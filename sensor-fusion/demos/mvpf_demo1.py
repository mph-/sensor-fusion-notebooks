# Michael P. Hayes UCECE, Copyright 2018--2019
import numpy as np
from ipywidgets import interact, fixed
from matplotlib.pyplot import subplots, show
from matplotlib.patches import Arc
from .lib.utils import wraptopi


class Beacon(object):

    def __init__(self, x, y, theta, num=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.num = num

    def plot(self, axes, marker='o', colour='blue', label=None, size=1,
             name=None):

        x, y, theta = self.x, self.y, self.theta

        xdx = size * np.cos(theta)
        xdy = size * np.sin(theta)
        ydx = size * np.cos(theta + np.pi/2)
        ydy = size * np.sin(theta + np.pi/2)

        axes.plot(x, y, marker, color=colour, label=label, markersize=10)

        axes.plot((x, x + xdx), (y, y + xdy), color='red', linewidth=3)
        axes.plot((x, x + ydx), (y, y + ydy), color='green', linewidth=3)
        if name is not None:
            axes.text(x + 0.5, y - 0.5, name)


def mvpf_demo1_plot(axes, beacon_x=15, beacon_y=8, beacon_theta=-75,
                    robot_x=3, robot_y=1, robot_theta=15):

    robot = Beacon(robot_x, robot_y, np.radians(robot_theta), 1)
    beacon = Beacon(beacon_x, beacon_y, np.radians(beacon_theta), 1)

    axes.clear()
    axes.grid(True)
    axes.axis('scaled')
    axes.set_xlim(-0.05, 20)
    axes.set_ylim(-0.05, 10)
    axes.set_xticks((0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20))

    robot.plot(axes, marker='p', colour='black', size=5, name='robot')
    beacon.plot(axes, name='beacon')

    r = np.sqrt((robot.x - beacon.x)**2 + (robot.y - beacon.y)**2)
    phi = np.arctan2((beacon.y - robot.y), (beacon.x - robot.x))

    phid = wraptopi(phi - robot.theta)

    axes.plot([robot.x, beacon.x], [robot.y, beacon.y], '--k')

    arc = Arc((robot.x, robot.y), 5, 5,
              theta1=np.degrees(robot.theta),
              theta2=np.degrees(phi))
    axes.add_patch(arc)

    axes.plot((0, 20), (0, 0), color='red', linewidth=3)
    axes.plot((0, 0), (0, 10), color='green', linewidth=3)

    axes.set_title('r=%.1f, phi=%.1f' % (r, np.degrees(phid)))


def mvpf_demo1():

    fig, axes = subplots(1, figsize=(10, 5))
    show()

    interact(mvpf_demo1_plot, axes=fixed(axes), robot_x=(0, 20, 1),
             robot_y=(0, 10, 1), robot_theta=(-180, 180, 15),
             beacon_x=(0, 20, 1), beacon_y=(0, 10, 1),
             beacon_theta=(-180, 180, 15), continuous_update=False)
