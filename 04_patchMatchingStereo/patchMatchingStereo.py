import taichi as ti
import numpy as np


INFINIT = 1e8
INTRINSIC_DIMENSION = 3
EXTRINSIC_DIMENSION = 4
RGB_IMAGE_DEPTH = 3
THREE_DIMENSION = 3


# config for debug can be optimzed
# currently, many shared data prevents parallel computing
# if enabel mult-threads, the cost will be significantly impacted.
ti.init(
    arch=ti.cuda,
    # debug=True,
    # kernel_profiler=True,
    # cpu_max_num_threads=1,
    # advanced_optimization=False
)


@ti.data_oriented
class patch_matching():
    def __init__(
        self,
        num_iters: int,
        patch_size: int,
        min_depth: int,
        max_depth: int,
        delta_depth: float,
        delta_norm: float,
        num_neighbors: int,
        num_refinement: int,
        likelihood_decay: float,
        balance_weight: float,
        tao_color: float,
        tao_gradient: float
    ) -> None:

        assert (patch_size+1) % 2 == 0, "patch size has to be an odd number"
        assert num_neighbors >= 2, "number of neighbors should >= 2"

        self.likelihood_decay: float = likelihood_decay
        self.balance_weight: float = balance_weight
        self.tao_color: float = tao_color
        self.tao_gradient: float = tao_gradient

        self.iterations: int = num_iters               # num of estimation iterations
        self.patch_size: int = patch_size              # size of each patch
        self.min_depth: int = min_depth                # min supported depth
        self.max_depth: int = max_depth                # max supported depth
        self.delta_depth: float = delta_depth          # step depth for refinement
        self.delta_norm: float = delta_norm            # step nomral for refinement
        self.num_neighbors: int = num_neighbors        # num of pixels for spacial prop
        self.num_refinement: int = num_refinement      # num of guesses for refinement

        # Patch matching input, output and intermediate data. These data
        # structures can be optimized via the advanced structure
        self.images: ti.field                          # input images
        self.depth_maps: ti.field                      # estimated depth maps
        self.normal_maps: ti.field                     # estimated normal maps
        self.cost_volumes: ti.field                    # computed cost maps
        self.src_image_idx: int                        # the idx of the src image
        self.intrinsics: ti.Matrix.field
        self.extrinsics: ti.Matrix.field

        # Resolution of each image
        self.resolutions: ti.Vector.field

        # Patch for cost calculation. They prevent the solution from
        # parallel processing on GPU. Can be set to Matrix if the patch
        # size is not very large and thus can be in parallel.
        self.src_patch: ti.field
        self.ref_patch: ti.field

        # Fields for propagation. Sharing these data prevent parallel
        # computing on GPU.
        self.normal_propagate_field: ti.Vector.field
        self.depth_propagate_field: ti.field

        # Debug purpose. Sharing these data prevent parallel
        # computing on GPU.
        self.patch_cost_debug = ti.field(dtype=ti.f32, shape=(
            self.patch_size, self.patch_size))
        self.patch_grad_debug = ti.field(dtype=ti.f32, shape=(
            self.patch_size, self.patch_size))

    def init_algorithm(
        self,
        images: list[np.ndarray],
        intrinsics: list[np.ndarray],
        extrinsics: dict,
        source_image_idx: int,
    ) -> None:

        ti.static_assert(len(images) == len(intrinsics), "Dimension error!")

        self.src_image_idx = source_image_idx

        # Intrinsics Matrix field, 3 dimension
        # [cam_idx, 3, 3]
        self.intrinsics = ti.Matrix.field(
            INTRINSIC_DIMENSION,
            INTRINSIC_DIMENSION,
            dtype=ti.f32,
            shape=len(intrinsics)
        )
        self.intrinsics.from_numpy(np.asarray(intrinsics, dtype='f'))

        # Extrinsics Matrix field, 4 dimension
        # [src_cam_idx, dst_cam_idx, 3, 4]
        self.extrinsics = ti.Matrix.field(
            EXTRINSIC_DIMENSION-1,
            EXTRINSIC_DIMENSION,
            dtype=ti.f32,
            shape=(len(images), len(images))
        )
        extrinsics_np = np.zeros(
            (len(images), len(images), EXTRINSIC_DIMENSION-1, EXTRINSIC_DIMENSION))
        for key in extrinsics.keys():
            extrinsics_np[key[0], key[1]] = extrinsics[key]
        self.extrinsics.from_numpy(extrinsics_np.astype('f'))

        # Resolution vector, [img_idx, 3]
        self.resolutions = ti.Vector.field(
            n=RGB_IMAGE_DEPTH,
            dtype=ti.i32,
            shape=len(intrinsics)
        )
        self.resolutions.from_numpy(
            np.array([img.shape for img in images]).astype('i'))

        # Image scalar field, same as numpy array shape
        # [img_idx, num_row, num_col, 3]
        self.images = ti.field(
            dtype=ti.f32,
            shape=np.asarray(images).shape
        )
        self.images.from_numpy(np.asarray(images).astype('f'))

        # Disparity maps with uniform distribution
        # [img_idx, num_row, num_col]
        self.depth_maps = ti.field(
            dtype=ti.f32,
            shape=np.asarray(images).shape[:-1]
        )
        random_depth = np.random.uniform(
            low=self.min_depth,
            high=self.max_depth,
            size=np.asarray(images).shape[:-1]
        )
        self.depth_maps.from_numpy(random_depth.astype('f'))

        # Normal maps with uniform distribution
        # [img_idx, num_row, num_col, 3]
        self.normal_maps = ti.field(
            dtype=ti.f32, shape=np.asarray(images).shape)
        random_normal = np.random.uniform(
            low=-1,
            high=1,
            size=np.asarray(images).shape
        )
        for i in range(random_normal.shape[0]):
            random_normal[i] = random_normal[i] / \
                np.linalg.norm(random_normal[i], axis=2)[:, :, None]
        self.normal_maps.from_numpy(random_normal.astype('f'))

        # Cost maps with INFINIT values
        # [img_idx, num_row, num_col]
        self.cost_volumes = ti.field(
            dtype=ti.f32, shape=images[0].shape[:-1])
        self.cost_volumes.from_numpy(
            np.ones(images[0].shape[:-1]).astype('f')*INFINIT)

        # Patch feild
        # The last dimension is set to the number of rows to
        # allow parallelization
        self.src_patch = ti.field(dtype=ti.f32, shape=(
            self.patch_size,
            self.patch_size,
            RGB_IMAGE_DEPTH,
            self.resolutions[self.src_image_idx].x)
        )
        self.ref_patch = ti.field(dtype=ti.f32, shape=(
            self.patch_size,
            self.patch_size,
            RGB_IMAGE_DEPTH,
            self.resolutions[self.src_image_idx].x)
        )

        # Propagation caches
        # The last dimension is set to the number of rows to
        # allow parallelization
        self.normal_propagate_field = ti.Vector.field(
            n=THREE_DIMENSION,
            dtype=ti.f32,
            shape=(
                1 + self.num_neighbors//2*2 + self.num_refinement,
                self.resolutions[self.src_image_idx].x
            )
        )
        self.depth_propagate_field = ti.field(
            dtype=ti.f32,
            shape=(
                1 + self.num_neighbors//2*2 + self.num_refinement,
                self.resolutions[self.src_image_idx].x
            )
        )

    @ti.kernel
    def calculate_cost_and_propagate(self, iteration: int):
        # As the initial implementation, only estimate the depth of the
        # first image. This is to prevent race condiction while parallelizing
        # the calculation and propagation.

        # for i in range(1499, 1500):
        #     for j in range(980, 981):
        for i in range(self.resolutions[self.src_image_idx].x):
            # Rech row has its own cache for intermediate results
            cache_index = i

            # At even iterations, traverse from left to right in parallel.
            # At odd iterations, traverse from right to left in parallel.
            start_idx = 0
            end_idx = self.resolutions[self.src_image_idx].y - 1

            if iteration % 2 == 1:
                start_idx = self.resolutions[self.src_image_idx].y - 1
                end_idx = 0

            for j in range(start_idx, end_idx):

                self.get_src_patch(i, j)

                # Get all candidates for propagation
                self.propagate_normal(i, j, iteration, cache_index)
                self.propagate_depth(i, j, iteration, cache_index)

                for prop_idx in range(self.normal_propagate_field.shape[0]):

                    cost_all_ref_views = 0.0
                    for ref_idx in range(self.images.shape[0]):
                        if ref_idx == self.src_image_idx:
                            continue

                        # print("calculate_cost_and_propagate--prop",
                        #     self.depth_propagate_field[prop_idx, cache_index],
                        #     self.normal_propagate_field[prop_idx, cache_index]
                        # )

                        self.get_ref_patch(
                            i,
                            j,
                            self.depth_propagate_field[prop_idx, cache_index],
                            self.normal_propagate_field[prop_idx, cache_index],
                            self.intrinsics[self.src_image_idx],
                            ref_idx
                        )

                        cost_all_ref_views += self.compute_cost(cache_index)

                    # If this is a better guess, update the cost, depth
                    # and the normal of the pixel.
                    if cost_all_ref_views < self.cost_volumes[i, j]:
                        self.cost_volumes[i, j] = cost_all_ref_views

                        self.depth_maps[self.src_image_idx, i, j] = \
                            self.depth_propagate_field[prop_idx, cache_index]

                        self.normal_maps[self.src_image_idx, i, j, 0] = \
                            self.normal_propagate_field[prop_idx,
                                                        cache_index].x
                        self.normal_maps[self.src_image_idx, i, j, 1] = \
                            self.normal_propagate_field[prop_idx,
                                                        cache_index].y
                        self.normal_maps[self.src_image_idx, i, j, 2] = \
                            self.normal_propagate_field[prop_idx,
                                                        cache_index].z

    @ti.func
    def propagate_depth(
        self,
        row: int,
        col: int,
        iters: int,
        patch_cache_index: int
    ) -> None:

        index = 0

        # Spacial propagation
        for i in range(-self.num_neighbors//2, self.num_neighbors//2+1):
            if ((col + i >= 0) and (col + i < self.resolutions[self.src_image_idx].y)):
                self.depth_propagate_field[index, patch_cache_index] = \
                    self.depth_maps[self.src_image_idx, row, col+i]
            else:
                self.depth_propagate_field[index, patch_cache_index] = (
                    self.depth_maps[self.src_image_idx, row, col] +
                    (ti.random(float) - 0.5)*self.delta_depth*ti.pow(2, -iters)
                )

                if self.depth_propagate_field[index, patch_cache_index] > self.max_depth:
                    self.depth_propagate_field[index,
                                               patch_cache_index] = self.max_depth

                if self.depth_propagate_field[index, patch_cache_index] < self.min_depth:
                    self.depth_propagate_field[index,
                                               patch_cache_index] = self.min_depth

            index += 1

        # Random refinement
        for i in range(self.num_refinement):
            self.depth_propagate_field[index, patch_cache_index] = (
                self.depth_propagate_field[self.num_neighbors//2, patch_cache_index] +
                (ti.random(float) - 0.5)*self.delta_depth*ti.pow(2, -iters)
            )

            if self.depth_propagate_field[index, patch_cache_index] > self.max_depth:
                self.depth_propagate_field[index,
                                           patch_cache_index] = self.max_depth

            if self.depth_propagate_field[index, patch_cache_index] < self.min_depth:
                self.depth_propagate_field[index,
                                           patch_cache_index] = self.min_depth

            index += 1

    @ti.func
    def propagate_normal(
        self,
        row: int,
        col: int,
        iters: int,
        patch_cache_index: int
    ) -> None:

        index = 0

        # Spacial propagation
        for i in range(-self.num_neighbors//2, self.num_neighbors//2+1):
            if ((col + i >= 0) and (col + i < self.resolutions[self.src_image_idx].y)):
                self.normal_propagate_field[index, patch_cache_index] = ti.Vector(
                    [self.normal_maps[self.src_image_idx, row, col+i, 0],
                     self.normal_maps[self.src_image_idx, row, col+i, 1],
                     self.normal_maps[self.src_image_idx, row, col+i, 2]]
                )
            else:
                self.normal_propagate_field[index, patch_cache_index] = ti.math.normalize(
                    ti.Vector(
                        [self.normal_maps[self.src_image_idx, row, col, 0],
                         self.normal_maps[self.src_image_idx, row, col, 1],
                         self.normal_maps[self.src_image_idx, row, col, 2]]) +
                    (ti.Vector([ti.random(float),
                                ti.random(float),
                                ti.random(float)]
                               ) - 0.5
                     )*self.delta_norm*ti.pow(2, -iters)
                )

            index += 1

        # Random refinement
        for i in range(self.num_refinement):
            self.normal_propagate_field[index, patch_cache_index] = ti.math.normalize(
                self.normal_propagate_field[self.num_neighbors//2, patch_cache_index] +
                (ti.Vector([ti.random(float),
                            ti.random(float),
                            ti.random(float)]
                           ) - 0.5
                 )*self.delta_norm*ti.pow(2, -iters)
            )
            index += 1

    @ti.func
    def get_src_patch(
        self,
        center_row: int,
        center_col: int
    ) -> None:

        patch_cache_index = center_row

        for i in range(self.patch_size):
            for j in range(self.patch_size):
                img_row = center_row + i - self.patch_size//2
                img_col = center_col + j - self.patch_size//2

                self.src_patch[i, j, 0, patch_cache_index] = 0.0
                self.src_patch[i, j, 1, patch_cache_index] = 0.0
                self.src_patch[i, j, 2, patch_cache_index] = 0.0

                if ((img_row >= 0) and (img_row < self.resolutions[self.src_image_idx].x)) and \
                        ((img_col >= 0) and (img_col <= self.resolutions[self.src_image_idx].y)):
                    self.src_patch[i, j, 0, patch_cache_index] = self.images[self.src_image_idx,
                                                                             img_row, img_col, 0]
                    self.src_patch[i, j, 1, patch_cache_index] = self.images[self.src_image_idx,
                                                                             img_row, img_col, 1]
                    self.src_patch[i, j, 2, patch_cache_index] = self.images[self.src_image_idx,
                                                                             img_row, img_col, 2]

    @ti.func
    def get_ref_patch(
        self,
        center_row_src: int,
        center_col_src: int,
        center_depth: float,
        center_normal: ti.template(),        # ti.types.ndarray() is for ti.fields data
        src_cam_intrinsic: ti.template(),    # ti,template() is for Vector and Matrix
        ref_image_idx: int
    ) -> None:
        # Calculate the plane constant
        fx = src_cam_intrinsic[0, 0]
        fy = src_cam_intrinsic[1, 1]
        cx = src_cam_intrinsic[0, 2]
        cy = src_cam_intrinsic[1, 2]
        Px = (center_col_src - cx)*center_depth/fx
        Py = (center_row_src - cy)*center_depth/fy
        P = ti.Vector([Px, Py, center_depth])
        C = center_normal.dot(P)

        for i in range(self.patch_size):
            for j in range(self.patch_size):
                img_row_src = center_row_src + i - self.patch_size//2
                img_col_src = center_col_src + j - self.patch_size//2

                self.ref_patch[i, j, 0, center_row_src] = 0.0
                self.ref_patch[i, j, 1, center_row_src] = 0.0
                self.ref_patch[i, j, 2, center_row_src] = 0.0

                if (((img_row_src >= 0) and
                     (img_row_src < self.resolutions[self.src_image_idx][0])
                     ) and
                        ((img_col_src >= 0) and
                         (img_col_src <
                          self.resolutions[self.src_image_idx][1])
                         )
                    ):
                    ref_pixel = self.project_pixel(
                        img_row_src, img_col_src, cx, cy, fx, fy, center_normal, C, ref_image_idx)

                    # print("get_ref_patch", img_row_src, img_col_src, ref_pixel)

                    if (((ref_pixel[1] >= 0) and
                             (ref_pixel[1] <
                              self.resolutions[self.src_image_idx][0])
                             ) and
                            ((ref_pixel[0] >= 0) and
                             (ref_pixel[0] <
                                      self.resolutions[self.src_image_idx][1])
                             )
                            ):
                        ref_color = self.interpolate_color(
                            ref_pixel, ref_image_idx)

                        self.ref_patch[i, j, 0, center_row_src] = ref_color.x
                        self.ref_patch[i, j, 1, center_row_src] = ref_color.y
                        self.ref_patch[i, j, 2, center_row_src] = ref_color.z

    @ti.func
    def project_pixel(
        self,
        q_row: int,
        q_col: int,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        center_normal: ti.template(),    # ti,template() is for Vector and Matrix
        plane_constant: float,
        ref_image_idx: int
    ) -> ti.Vector:
        Qx_bar = (q_col - cx)/fx
        Qy_bar = (q_row - cy)/fy

        Qz = plane_constant / \
            (center_normal.z + center_normal.x*Qx_bar + center_normal.y*Qy_bar)
        Qx = Qx_bar*Qz
        Qy = Qy_bar*Qz
        Q = ti.Vector([Qx, Qy, Qz, 1])

        Q_ref = self.extrinsics[self.src_image_idx, ref_image_idx]@Q
        Q_ref_fp = Q_ref/Q_ref.z
        q_ref = self.intrinsics[ref_image_idx]@Q_ref_fp

        q_ref_row = 0.0
        q_ref_col = 0.0
        if (((q_ref.x >= 0) and
                 (q_ref.x < self.resolutions[self.src_image_idx][1])) and
                    ((q_ref.y >= 0) and
                     (q_ref.y < self.resolutions[self.src_image_idx][0]))
                ):
            q_ref_row = q_ref.y
            q_ref_col = q_ref.x

        q_ref_pixel = ti.Vector([q_ref_row, q_ref_col])

        return q_ref_pixel

    @ti.func
    def interpolate_color(
        self,
        ref_pixel: ti.template(),
        ref_image_idx: int
    ) -> ti.Vector:

        row1 = int(ref_pixel[0] - 0.5)
        row2 = int(ref_pixel[0] + 0.5)
        col1 = int(ref_pixel[1] - 0.5)
        col2 = int(ref_pixel[1] + 0.5)

        # print("interpolate_color", ref_pixel, col1, col2, row1, row2)

        color_tl = ti.Vector([self.images[ref_image_idx, row1, col1, 0],
                              self.images[ref_image_idx, row1, col1, 1],
                              self.images[ref_image_idx, row1, col1, 2]])
        color_tr = ti.Vector([self.images[ref_image_idx, row1, col2, 0],
                              self.images[ref_image_idx, row1, col2, 1],
                              self.images[ref_image_idx, row1, col2, 2]])
        color_bl = ti.Vector([self.images[ref_image_idx, row2, col1, 0],
                              self.images[ref_image_idx, row2, col1, 1],
                              self.images[ref_image_idx, row2, col1, 2]])
        color_br = ti.Vector([self.images[ref_image_idx, row2, col2, 0],
                              self.images[ref_image_idx, row2, col2, 1],
                              self.images[ref_image_idx, row2, col2, 2]])

        common_factor = 1.0/((col2 - col1)*(row2 - row1))
        w_tl = (col2 - ref_pixel[1])*(row2 - ref_pixel[0])
        w_tr = (ref_pixel[1] - col1)*(row2 - ref_pixel[0])
        w_bl = (col2 - ref_pixel[1])*(ref_pixel[0] - row1)
        w_br = (ref_pixel[1] - col1)*(ref_pixel[0] - row1)

        # print("interpolate_color", common_factor, w_tl, w_tr, w_bl, w_br)

        ref_color = (color_tl*w_tl + color_tr*w_tr +
                     color_bl*w_bl + color_br*w_br)*common_factor

        return ref_color

    @ti.func
    def compute_cost(self, patch_cache_index: int) -> float:

        cost_sum = 0.0
        src_center_clr = self.get_patch_pixel(
            self.src_patch,
            patch_cache_index,
            self.patch_size//2,
            self.patch_size,
            self.patch_size//2,
            self.patch_size
        )

        # print("compute_cost--cent_clr", src_center_clr)

        for i in range(self.patch_size):
            for j in range(self.patch_size):
                src_pixel_clr = self.get_patch_pixel(
                    self.src_patch, patch_cache_index, i, self.patch_size, j, self.patch_size)
                ref_pixel_clr = self.get_patch_pixel(
                    self.ref_patch, patch_cache_index, i, self.patch_size, j, self.patch_size)

                # print("compute_cost--src_clr, ref_clr:",
                #       src_pixel_clr, ref_pixel_clr)

                likelihood = self.compute_likelihood(
                    src_center_clr, src_pixel_clr)

                clr_cost = self.compute_color_cost(
                    src_pixel_clr, ref_pixel_clr)

                grd_cost = self.compute_gradiant_cost(i, j, patch_cache_index)

                cost = likelihood * \
                    ((1 - self.balance_weight)*clr_cost +
                     self.balance_weight*grd_cost)

                # self.patch_cost_debug[i, j] = cost

                cost_sum += cost

                # print("compute_cost--costs:", likelihood,
                #       clr_cost, grd_cost, cost)
        return cost_sum

    @ti.func
    def compute_likelihood(self, color_p, color_q) -> float:
        pwr = - ti.abs(color_p - color_q).sum()/self.likelihood_decay
        likelihood = ti.math.exp(pwr)
        return likelihood

    @ti.func
    def compute_color_cost(self, color_q, color_q_prime) -> float:
        color_dissimilarity = ti.abs(color_q - color_q_prime).sum()
        ti.atomic_min(color_dissimilarity, self.tao_color)
        return color_dissimilarity

    @ti.func
    def compute_gradiant_cost(self, i: int, j: int, patch_cache_index: int) -> float:
        src_tl = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i-1, self.patch_size, j-1, self.patch_size)
        src_tp = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i-1, self.patch_size, j, self.patch_size)
        src_tr = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i+1, self.patch_size, j-1, self.patch_size)
        src_bl = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i+1, self.patch_size, j-1, self.patch_size)
        src_bt = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i+1, self.patch_size, j, self.patch_size)
        src_br = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i+1, self.patch_size, j+1, self.patch_size)
        src_lf = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i, self.patch_size, j-1, self.patch_size)
        src_rg = self.get_patch_pixel(
            self.src_patch, patch_cache_index, i, self.patch_size, j+1, self.patch_size)

        ref_tl = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i-1, self.patch_size, j-1, self.patch_size)
        ref_tp = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i-1, self.patch_size, j, self.patch_size)
        ref_tr = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i-1, self.patch_size, j+1, self.patch_size)
        ref_bl = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i+1, self.patch_size, j-1, self.patch_size)
        ref_bt = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i+1, self.patch_size, j, self.patch_size)
        ref_br = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i+1, self.patch_size, j+1, self.patch_size)
        ref_lf = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i, self.patch_size, j-1, self.patch_size)
        ref_rg = self.get_patch_pixel(
            self.ref_patch, patch_cache_index, i, self.patch_size, j+1, self.patch_size)

        # Get x-gradient
        src_grad_x = (src_tr - src_tl) + 2 * \
            (src_rg - src_lf) + (src_br - src_bl)
        ref_grad_x = (ref_tr - ref_tl) + 2 * \
            (ref_rg - ref_lf) + (ref_br - ref_bl)

        # Get y-gradient
        src_grad_y = (src_tl - src_bl) + 2 * \
            (src_tp - src_bt) + (src_tr - src_br)
        ref_grad_y = (ref_tl - ref_bl) + 2 * \
            (ref_tp - ref_bt) + (ref_tr - ref_br)

        # Get square root of sum of squares
        src_grad = ti.sqrt(src_grad_x*src_grad_x + src_grad_y*src_grad_y)
        ref_grad = ti.sqrt(ref_grad_x*ref_grad_x + ref_grad_y*ref_grad_y)

        # self.patch_grad_debug[i, j] = ref_grad.sum()

        grad_dissimilarity = ti.abs(src_grad - ref_grad).sum()
        ti.atomic_min(grad_dissimilarity, self.tao_gradient)

        return grad_dissimilarity

    @ti.func
    def get_patch_pixel(
        self,
        image: ti.template(),
        patch_cache_index: int,
        i: int, max_i: int,
        j: int, max_j: int
    ) -> ti.Vector:

        pixel = ti.Vector([0.0, 0.0, 0.0])
        if (((i >= 0) and (i < max_i)) and ((j >= 0) and (j < max_j))):
            pixel = ti.Vector([image[i, j, 0, patch_cache_index],
                               image[i, j, 1, patch_cache_index],
                               image[i, j, 2, patch_cache_index]])
        return pixel

    def run(
        self,
        images: list[np.ndarray],
        intrinsics: list[np.ndarray],
        extrinsics: dict,
        source_image_idx: int,
        gui_enable: bool = False
    ) -> None:

        self.init_algorithm(
            images,
            intrinsics,
            extrinsics,
            source_image_idx,
        )

        for i in range(self.iterations):
            print("Iteration {} starts".format(i))

            if gui_enable:
                self.visualize_fields(i)

            self.calculate_cost_and_propagate(i)

    def visualize_fields(self, iteration_num: int):
        pass
