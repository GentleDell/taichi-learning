from os import path
from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt

from patchMatchingStereo import patch_matching as pm


# Patch Matching Parameters
PATCH_SIZE = 35
ITERATIONS = 3
NUM_REFINEMENT_GUESS = 2
NUM_SPACIAL_PROPERGATION = 2    # apply to both left/right and top/bottom
MIN_DEPTH = 0.1
MAX_DEPTH = 5
DELTA_NORMAL= 0.2
LIKELIHOOD_DECAY = 10
BALANCE_WEIGHT = 0.9
TAO_COLOR = 10
TAO_GRADIENT = 2

# Data Infos
DATASET = "./04_patchMatchingStereo/Piano-perfect"
SOURCE_IMAGE_IDX = 0


class data_parser():
    def __init__(self, path_to_dataset: str):

        assert path.isdir(path_to_dataset), "Invalid path"

        image0 = path.join(path_to_dataset, "im0.png")
        assert path.isfile(image0), "No image 0"

        image1 = path.join(path_to_dataset, "im1.png")
        assert path.isfile(image1), "No image 1"

        self.images = [plt.imread(img) for img in (image0, image1)]
        self.camera_intrinsics: list[np.ndarray] = []
        self.camera_extrinsics: dict = {}
        self.min_disparity: float
        self.max_disparity: float

        baseline = 0
        ndisp = 0
        with open(path.join(path_to_dataset, "calib.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if "cam" in line:
                    self.camera_intrinsics.append(np.matrix(line.split("=")[1], dtype='f'))
                if "baseline" in line:
                    baseline = float(line.split("=")[1])/1000
                    self.camera_extrinsics[(0, 1)] = np.array([
                        [1, 0, 0, -baseline],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]],
                        dtype='f'
                    )
                    self.camera_extrinsics[(1, 0)] = self.camera_extrinsics[(0, 1)]
                if "vmin" in line:
                    self.min_disparity = float(line.split("=")[1])
                if "vmax" in line:
                    self.max_disparity = float(line.split("=")[1])  
                if "ndisp" in line:
                    # important to have correct disparity
                    ndisp = float(line.split("=")[1])

        self.gt_depth = []
        gt_disparity0 = path.join(path_to_dataset, "disp0.pfm")
        gt_disparity1 = path.join(path_to_dataset, "disp1.pfm")
        for d_file in (gt_disparity0, gt_disparity1):
            disparity = self.read_pfm(d_file)
            depth = baseline * self.camera_intrinsics[0][0, 0] / (
                disparity*ndisp + self.camera_intrinsics[1][0, 2] - self.camera_intrinsics[0][0, 2])
            self.gt_depth.append(depth)


    def read_pfm(self, filename):
        with Path(filename).open('rb') as pfm_file:

            line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
            assert line1 in ('PF', 'Pf')
            
            channels = 3 if "PF" in line1 else 1
            width, height = (int(s) for s in line2.split())
            scale_endianess = float(line3)
            bigendian = scale_endianess > 0
            scale = abs(scale_endianess)

            buffer = pfm_file.read()
            samples = width * height * channels
            assert len(buffer) == samples * 4
            
            fmt = f'{"<>"[bigendian]}{samples}f'
            decoded = struct.unpack(fmt, buffer)
            shape = (height, width, 3) if channels == 3 else (height, width)
            return np.flipud(np.reshape(decoded, shape)) * scale

def main():

    dataset = data_parser(DATASET)

    pm_solver = pm(
        num_iters=ITERATIONS,
        patch_size=PATCH_SIZE,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        delta_depth=(MAX_DEPTH-MIN_DEPTH)/2,
        delta_norm=DELTA_NORMAL,
        num_neighbors=NUM_SPACIAL_PROPERGATION,
        num_refinement=NUM_REFINEMENT_GUESS,
        likelihood_decay=LIKELIHOOD_DECAY,
        balance_weight=BALANCE_WEIGHT,
        tao_color=TAO_COLOR,
        tao_gradient=TAO_GRADIENT
    )

    pm_solver.run(
        images=dataset.images,
        intrinsics=dataset.camera_intrinsics,
        extrinsics=dataset.camera_extrinsics,
        source_image_idx=SOURCE_IMAGE_IDX,
        gui_enable=True
    )
    
    # Copy data to cpu numpy for plt.imshow is slow
    depth_maps = pm_solver.depth_maps.to_numpy()[SOURCE_IMAGE_IDX]
    normal_maps= pm_solver.normal_maps.to_numpy()[SOURCE_IMAGE_IDX]
    cost_volumes=pm_solver.cost_volumes.to_numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(depth_maps, vmin=MIN_DEPTH, vmax=MAX_DEPTH)
    ax1.title.set_text('depth_maps')
    ax2 = fig.add_subplot(222)
    ax2.imshow((normal_maps+1)/2, vmin=0, vmax=1)
    ax2.title.set_text('normal_maps')
    ax3 = fig.add_subplot(223)
    ax3.imshow(cost_volumes, vmin=cost_volumes.min(), vmax=PATCH_SIZE**2)
    ax3.title.set_text('cost_volumes')
    ax4 = fig.add_subplot(224)
    ax4.imshow(dataset.images[SOURCE_IMAGE_IDX])
    ax4.title.set_text('src image')
    plt.show(block=False)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(dataset.gt_depth[SOURCE_IMAGE_IDX],
               vmin=MIN_DEPTH, vmax=MAX_DEPTH)
    ax1.title.set_text('ground truth depth')
    ax2 = fig.add_subplot(122)
    ax2.imshow(
        np.abs(dataset.gt_depth[SOURCE_IMAGE_IDX] - depth_maps), vmin=0, vmax=MAX_DEPTH)
    ax2.title.set_text('depth residual error')
    plt.show(block=False)

    residual = np.abs(dataset.gt_depth[SOURCE_IMAGE_IDX] - depth_maps).mean()
    print("Residual depth error", residual)

    input("Press any key to stop.")

if __name__ == '__main__':
    main()
    