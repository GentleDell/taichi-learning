import taichi as ti
from celestial_object import Planet, Star
import time

ti.init(ti.cuda)

paused = False

star_mass = [1000] * 4
star_size = [8] * 4
planet_mass = [1] * 1000
planet_size = [2] * 1000

stars = Star(star_mass, star_size)
stars.initialize(0.5, 0.5, 0.2, 10)
planets = Planet(planet_mass, planet_size)
planets.initialize(0.5, 0.5, 0.4, 3)

my_gui = ti.GUI("Galaxy", (800, 800))
h = 5e-5  # time-step size
i = 0
while my_gui.running:
    for e in my_gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            paused = not paused
            print("paused =", paused)
        elif e.key == "r":
            pass
        elif e.key == "b":
            pass
        elif e.key == ti.GUI.UP:
            pass
        elif e.key == ti.GUI.DOWN:
            pass

    if not paused:
        for obj in (stars, planets):
            obj.clearAcc()

        
        stars.computeInternalAcc()
        stars.computeAcc(planets)
        planets.computeInternalAcc()
        planets.computeAcc(stars)

        for obj in (stars, planets):
            obj.update(h)

        i += 1

    stars.display(my_gui, color=0xffd500)
    planets.display(my_gui)

    my_gui.text(
        content="frame: {}".format(i), pos=(0, 1.0), color=0xFFFFFF)
    my_gui.text(
        content="b: Add/Remove a Super Star (100 times of star mass)", pos=(0, 0.92), color=0xFFFFFF)
    my_gui.text(
        content="r: Reset", pos=(0, 0.98), color=0xFFFFFF)
    my_gui.text(
        content="i: Save to Image (not working, disabled)", pos=(0, 0.96), color=0xFFFFFF)
    # my_gui.text(
    #     content=f"Up/Down: Increase/Decrease planet mass ({planets.GetMass()/1000.0:.05f} times of star mass)", pos=(0, 0.94), color=0xFFFFFF)
    # my_gui.text(
    #     content=f"Space: Pauss/Continue", pos=(0, 0.92), color=0xFFFFFF)

    my_gui.show()
    