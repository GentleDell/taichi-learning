import taichi as ti
from celestial_objects import Star, Planet, SuperStar

if __name__ == "__main__":

    ti.init(arch=ti.cuda)

    # control
    paused = False
    export_images = False

    # stars and planets
    stars = Star(N=2, mass=1000)
    stars.initialize(0.5, 0.5, 0.2, 10)
    planets = Planet(N=1000, mass=1)
    planets.initialize(0.5, 0.5, 0.4, 3)

    superstars = SuperStar(N=1, mass=100000)
    superstars.initialize(0.6, 0.6, 0.0, 0)

    # GUI
    my_gui = ti.GUI("Galaxy", (800, 800))
    h = 5e-5 # time-step size
    i = 0
    has_superstar = False
    while my_gui.running:
        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            elif e.key == 'r':
                stars.initialize(0.5, 0.5, 0.2, 10)
                planets.initialize(0.5, 0.5, 0.4, 10)
                planets.SetMass(1)
                has_superstar = False
                i = 0
            elif e.key == 'b':
                if (not has_superstar):
                    has_superstar = True
                else:
                    has_superstar = False
            elif e.key == 'i':
                # export_images = not export_images
                pass # disabled, for it will cause some error in dll
            elif e.key == ti.GUI.UP:
                planets.SetMass(planets.GetMass() * 1.1 )
            elif e.key == ti.GUI.DOWN:
                planets.SetMass(planets.GetMass() / 1.1)
            

        if not paused:
            for obj in (stars, planets):
                obj.clearForce()

            if (has_superstar):
                stars.computeForce(superstars)
                planets.computeForce(superstars)
            stars.computeForceInternal()
            planets.computeForce(stars)
            stars.computeForce(planets)
            for celestial_obj in (stars, planets):
                celestial_obj.update(h)
            i += 1

        if (has_superstar):
            superstars.display(my_gui, radius=20, color=0x472539)
        stars.display(my_gui, radius=10, color=0xffd500)
        planets.display(my_gui)
        my_gui.text(
            content=f'b: Add/Remove a Super Star (100 times of star mass)', pos=(0, 0.92), color=0xFFFFFF)
        my_gui.text(
            content=f'r: Reset', pos=(0, 0.98), color=0xFFFFFF)
        my_gui.text(
            content=f'i: Save to Image (not working, disabled)', pos=(0, 0.96), color=0xFFFFFF)
        my_gui.text(
            content=f'Up/Down: Increase/Decrease planet mass ({planets.GetMass()/1000.0:.05f} times of star mass)', pos=(0, 0.94), color=0xFFFFFF)
        my_gui.text(
            content=f'Space: Pauss/Continue', pos=(0, 0.92), color=0xFFFFFF)


        if export_images:
            my_gui.show(f"images\output_{i:05}.png")
        else:
            my_gui.show()