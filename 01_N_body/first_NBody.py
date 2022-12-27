import taichi as ti
import numpy as np

ti.init(ti.gpu)

###########################
######## Data Part ########
###########################

G_constant = 6.67432e-1       # m^3/(kg*s^2)
PI = 3.1415926535

num_body = 100
size_gala = 0.4               # percent of drawing
WindowSize = 512              # pixel

# pixels
radius_body_np = np.array([6] + [6] + [2] * (num_body - 2))          
radius_body_ti = ti.field(ti.f32, shape=num_body)
radius_body_ti.from_numpy(radius_body_np) 

# kg
mass_body_np = np.array([1000] + [1000] + [10] * (num_body - 2))           
mass_body_ti = ti.field(ti.i32, shape=num_body)
mass_body_ti.from_numpy(mass_body_np)

# m/s
init_vel_np = np.array([120] + [120] + [120] * (num_body - 2))           
init_vel_ti = ti.field(ti.f32, shape=num_body)
init_vel_ti.from_numpy(init_vel_np)

timestep = 1e-5                # gui step, in second
subtstep = 10                  # update steps to get gui step

pos = ti.Vector.field(n=2, dtype=ti.f32, shape=num_body)
vel = ti.Vector.field(n=2, dtype=ti.f32, shape=num_body)
acc = ti.Vector.field(n=2, dtype=ti.f32, shape=num_body)


#############################
######## Computation ########
#############################

@ti.kernel
def initialization():
    center_gala = ti.Vector([0.5, 0.5])
    for i in range(num_body):
        if i <= 1:
            theta = ti.random() * 2 * PI
            distance =  (ti.random() * 0.4 + 0.3) * size_gala
            offset = distance * ti.Vector([ti.cos(theta), ti.sin(theta)])
            pos[i] = center_gala + offset
            vel[i] = [-offset.y, offset.x]
            vel[i] *= init_vel_ti[i]
        else:
            theta = ti.random() * 2 * PI
            distance = (ti.random() * 0.7 + 0.3) * size_gala
            offset = distance * ti.Vector([ti.cos(theta), ti.sin(theta)])
            pos[i] = center_gala + offset
            vel[i] = [-offset.y, offset.x]
            vel[i] *= init_vel_ti[i]


@ti.kernel
def compute_acc():
    # clear data
    for i in range(num_body):
        acc[i] = ti.Vector([0.0, 0.0])

    # compute acceleration
    for i in range(num_body):
        p = pos[i]
        for j in range(num_body):

            distance_vec = p - pos[j]
            r = distance_vec.norm(1e-2)

            force = G_constant * mass_body_ti[i] * \
                mass_body_ti[j] * (1.0/r)**3 * distance_vec

            acc[i] += -force/(mass_body_ti[i]+1e-5)
            acc[j] += force/(mass_body_ti[j]+1e-5)


@ti.kernel
def update():
    dt = timestep/subtstep

    for i in range(num_body):
        vel[i] += dt*acc[i]
        pos[i] += dt*vel[i]

@ti.kernel
def merge():
    for i in range(num_body - 1):
        p = pos[i]
        for j in range(i+1, num_body):

            if mass_body_ti[i] <= 0 or mass_body_ti[j] <= 0:
                continue

            distance_vec = p - pos[j]
            r = distance_vec.norm(1e-5)
            if r <= ((radius_body_ti[i] + radius_body_ti[j])/WindowSize/1.2):
                active_body = -1
                passive_body = -1
                if mass_body_ti[i] >= mass_body_ti[j]:
                    active_body = i
                    passive_body = j
                else:
                    active_body = j 
                    passive_body = i
                    
                mass_body_ti[active_body] += mass_body_ti[passive_body]
                mass_body_ti[passive_body] = 0
                pos[passive_body] = [0., 0.]
                vel[passive_body] = [0., 0.]
                acc[passive_body] = [0., 0.]
                # radius_body_ti[passive_body] = 1

###############################
######## Visualization ########
###############################

gui = ti.GUI(name="{}-body problem".format(num_body),
             res=(WindowSize, WindowSize))

initialization()
count = 0
while gui.running:

    for i in range(subtstep):
        compute_acc()
        update()
        merge()
        
    gui.clear(color=0x112F41)
    gui.circles(pos.to_numpy(), color=0xFFFFFF, radius=radius_body_np)
    gui.show()
