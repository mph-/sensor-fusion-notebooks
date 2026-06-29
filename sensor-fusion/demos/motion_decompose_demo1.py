# Michael P. Hayes UCECE, Copyright 2018--2019
from numpy import degrees, radians, cos, sin, arctan2, sqrt, tan, pi
from matplotlib.pyplot import subplots
from ipywidgets import interact, fixed
from .lib.robot import robot_draw
from .lib.pose import Pose


def motion_model(pose0, v, omega, dt):

    x0, y0, theta0 = pose0

    if omega == 0.0:
        x1 = x0 + v * cos(theta0) * dt
        y1 = y0 + v * sin(theta0) * dt
        theta1 = theta0
    else:
        x1 = x0 - v / omega * sin(theta0) + v / omega * sin(theta0 + omega * dt)
        y1 = y0 + v / omega * cos(theta0) - v / omega * cos(theta0 + omega * dt)
        theta1 = theta0 + omega * dt

    return (x1, y1, theta1)


def odom_decompose(pose1, pose0):

    x1, y1, p1 = pose0
    x2, y2, p2 = pose1

    phi1 = arctan2(y2 - y1, x2 - x1) - p1
    d = sqrt((y2 - y1)**2 + (x2 - x1)**2)
    phi2 = p2 - p1 - phi1

    return phi1, d, phi2


def speeds_decompose(pose1, pose0, dt):

    x1, y1, p1 = pose0
    x2, y2, p2 = pose1
    omega = (pose1[2] - pose0[2]) / dt

    d = sqrt((y2 - y1)**2 + (x2 - x1)**2)

    if omega == 0:
        v = d / dt
    else:
        v = omega * d / (2 * tan(omega * dt / 2))
    return v, omega


def motion_decompose_demo1_plot(x0=0, y0=0, theta0=0, v=2, omega=0):

    dt = 1.0
    pose0 = (x0, y0, radians(theta0))
    pose1 = motion_model(pose0, v, radians(omega), dt)

    phi1, d, phi2 = odom_decompose(pose1, pose0)

    v, omega = speeds_decompose(pose1, pose0, 1.0)

    fig, ax = subplots(figsize=(10, 5))
    Pose(0, 0, 0).draw_axes(ax)

    ax.axis('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True)

    robot_draw(ax, pose0[0], pose0[1], pose0[2], colour='blue')
    robot_draw(ax, pose1[0], pose1[1], pose1[2], colour='orange')

    ax.set_title('$\phi_1 = %.1f$ deg, $d = %.1f$ m, $\phi_2 = %.1f$ deg' %
                 (degrees(phi1), d, degrees(phi2)))


def motion_decompose_demo1():
    interact(motion_decompose_demo1_plot,
             x0=(-2, 2, 0.1), y0=(-2, 2, 0.1), theta0=(-180, 180, 15),
             v=(0, 2, 0.1), omega=(-60, 60, 15),
             continuous_update=False)
