from taichi.examples.patterns import taichi_logo
import matplotlib.pyplot as plt

import taichi as ti

ti.init(arch=ti.cpu)

n = 512
x = ti.field(dtype=ti.i32)              # entry point to the sturcture
res = n + n // 4 + n // 16 + n // 64    # leave some room for scatters
img = ti.field(dtype=ti.f32, shape=(res, res))

# Can be designed from block3 to block1 
block1 = ti.root.pointer(ti.ij, n // 64)# the root points to a 8x8 block1
block2 = block1.pointer(ti.ij, 4)       # each block1 represents 4x4 block2
block3 = block2.pointer(ti.ij, 4)       # each block2 represents 4x4 block3
block3.dense(ti.ij, 4).place(x)         # each block3 represents 4x4 pixels


@ti.kernel
def activate(t: ti.f32):
    for i, j in ti.ndrange(n, n):
        # get image coordinate (0->1).
        p = ti.Vector([i, j]) / n
        
        # convert to image center coordinate, transform then pack to the
        # image coordinate.
        p = ti.math.rotation2d(ti.sin(t)) @ (p - 0.5) + 0.5

        if taichi_logo(p) == 0:
            x[i, j] = 1


@ti.func
def scatter(i):
    # img is 680x680. During painting, the i and j range from 0 to 511,
    # result in max 676. So, leave 2 lines of pixels on the boundary.
    return i + i // 4 + i // 16 + i // 64 + 2


@ti.kernel
def paint():
    for i, j in ti.ndrange(n, n):
        t = x[i, j]
        block1_index = ti.rescale_index(x, block1, [i, j])    # get the index of x[i,j] at the level of block1
        block2_index = ti.rescale_index(x, block2, [i, j])    # get the index of x[i,j] at the level of block2
        block3_index = ti.rescale_index(x, block3, [i, j])    # get the index of x[i,j] at the level of block3

        # cumulate the activation status of ancestors
        t += ti.is_active(block1, block1_index)
        t += ti.is_active(block2, block2_index)
        t += ti.is_active(block3, block3_index)

        # map deactivated cells to 1 and others decresing to 0
        img[scatter(i), scatter(j)] = 1 - t / 4


def main():
    img.fill(0.05)

    gui = ti.GUI('Sparse Grids', (res, res))

    for i in range(100000):
        block1.deactivate_all()
        activate(i * 0.05)
        paint()
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
    