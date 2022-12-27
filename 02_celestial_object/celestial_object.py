import taichi as ti
import numpy as np

G_constant = 1       # m^3/(kg*s^2)
PI = 3.1415926535


@ti.data_oriented
class celestial_object():
    def __init__(self, mass: list[int], radius: list[int]) -> None:
        assert(len(mass) == len(radius))
        self.num = len(mass)
        self.radius = ti.Vector(radius)
        self.mass = ti.field(ti.i32, shape=self.num)
        self.mass.from_numpy(np.array(mass))

        self.pos = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)
        self.vel = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)
        self.acc = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)

    @ti.kernel
    def initialize(self, center_x: ti.f32, center_y: ti.f32, zoom: ti.f32, init_vel: ti.f32):
        for i in range(self.num):
            if self.num == 1:
                self.pos[i] = ti.Vector([center_x, center_y])
                self.vel[i] = ti.Vector([0.0, 0.0])
            else:
                theta, r = self.generateThetaR(i)
                offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)]) * zoom
                origin = ti.Vector([center_x, center_y])
                self.pos[i] = origin + offset
                self.vel[i] = [-offset.y, offset.x]
                self.vel[i] *= init_vel

    @ti.kernel
    def clearAcc(self):
        for i in self.acc:
            self.acc[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def computeInternalAcc(self):
        for i in range(self.num-1):
            p = self.pos[i]
            for j in range(i+1, self.num):
                distance_vec = p - self.pos[j]
                r = distance_vec.norm(1e-2)

                force = G_constant * self.mass[i] * \
                    self.mass[j] * distance_vec / r**3

                self.acc[i] += -force/(self.mass[i]+1e-5)
                self.acc[j] += force/(self.mass[j]+1e-5)

    @ti.kernel
    def computeAcc(self, external_obj: ti.template()):
        for i in range(self.num):
            p = self.pos[i]
            for j in range(external_obj.num):
                distance_vec = p - external_obj.pos[j]
                # The value in the norm() impacts how the planets behave when
                # two planets are close to each other by limiting the r norm.
                # Since the r will be divided to get the force, a r close to 0 
                # brings ill force.
                r = distance_vec.norm(1e-2)

                force = G_constant * self.mass[i] * \
                    external_obj.mass[j] * distance_vec / r**3

                self.acc[i] += -force/(self.mass[i]+1e-5)
                external_obj.acc[j] += force/(external_obj.mass[j]+1e-5)

    @ti.kernel
    def update(self, dt: ti.f32):
        for i in self.vel:
            self.vel[i] += dt*self.acc[i]
            self.pos[i] += dt*self.vel[i]

    def display(self, gui, color=0xffffff):
        gui.circles(self.pos.to_numpy(),
                    radius=self.radius.to_numpy(), color=color)

    def generateThetaR(self, i: ti.int16):
        "To be overwritten"
        pass


@ti.data_oriented
class Star(celestial_object):
    def __init__(self, mass: list[int], radius: list[int]) -> None:
        super().__init__(mass, radius)

    @ti.func
    def generateThetaR(self, i: ti.int16):
        theta = 2*PI*i/ti.cast(self.num, ti.f32)
        r = 1
        return theta, r


@ti.data_oriented
class Planet(celestial_object):
    def __init__(self, mass: list[int], radius: list[int]) -> None:
        super().__init__(mass, radius)

    @ti.func
    def generateThetaR(self, i: ti.int16):
        theta = 2 * PI * ti.random()  # theta \in (0, 2PI)
        r = (ti.sqrt(ti.random()) * 0.4 + 0.6)  # r \in (0.6,1)
        return theta, r
