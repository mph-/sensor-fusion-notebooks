# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from matplotlib.pyplot import arrow
from ipywidgets import interact, fixed, interactive, fixed
from matplotlib.pyplot import subplots
from numpy.random import randn, uniform, seed
from .lib.robot import robot_draw, Robot
from .lib.pose import Pose

def particles_motion_model_demo2_plot(Xmin=-1, Xmax=1, Ymin=0, Ymax=1,
                                      ThetaMin=90, ThetaMax=90, Nparticles=10,
                                      v=1, omega=0, steps=0, sigmaV=0,
                                      sigmaOmega=0):

    ThetaMin = np.radians(ThetaMin)
    ThetaMax = np.radians(ThetaMax)    

    seed(1)
    
    robots = []
    for m in range(Nparticles):
        robot = Robot(uniform(Xmin, Xmax), uniform(Ymin, Ymax),
                      uniform(ThetaMin, ThetaMax))
        robots.append(robot)


    fig, ax = subplots(figsize=(10, 5))        
    Pose(0, 0, 0).draw_axes(ax)        

    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 5)
    ax.grid(True)

    for n in range(steps + 1):
        colour = ['red', 'orange', 'green', 'blue', 'magenta'][n % 5]
        
        for m, robot in enumerate(robots):
            robot_draw(ax, robot.x, robot.y, robot.heading, colour=colour)
            v1 = v + np.random.randn() * sigmaV
            omega1 = omega + np.random.randn() * sigmaOmega
            
            robot.transition(v1, omega1, dt=1)
    

def particles_motion_model_demo2():
    interact(particles_motion_model_demo2_plot,
             v=(0, 2, 0.1), omega=(-2, 2, 0.1),
             Xmin=(-1, 1, 0.1), Xmax=(-1, 1, 0.1),
             Ymin=(-1, 1, 0.1), Ymax=(-1, 1, 0.1),
             ThetaMin=(-180, 180, 15), ThetaMax=(-180, 180, 15),
             Nparticles=(10, 100, 10),
             sigmaV=(0, 1, 0.1), sigmaOmega=(0, 1, 0.05),
             steps=(0, 5),
             continuous_update=False)
    
