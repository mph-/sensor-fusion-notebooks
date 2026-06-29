# Michael P. Hayes UCECE, Copyright 2018--2019
from matplotlib.pyplot import subplots, show
from ipywidgets import interact, fixed
from matplotlib.patches import Arc
from .lib.robot import robot_draw
from .lib.pose import Pose
from numpy import degrees, radians, cos, sin, arctan2, sqrt, pi


def motion_model(pose0, v, omega, dt):

    x0, y0, theta0 = pose0

    if omega == 0.0:
        x1 = x0 + v * cos(theta0) * dt
        y1 = y0 + v * sin(theta0) * dt
        theta1 = theta0
    else:
        x1 = x0 - v / omega * sin(theta0) + \
            v / omega * sin(theta0 + omega * dt)
        y1 = y0 + v / omega * cos(theta0) - \
            v / omega * cos(theta0 + omega * dt)
        theta1 = theta0 + omega * dt

    return (x1, y1, theta1)


def odom_decompose(pose1, pose0):

    x1, y1, p1 = pose0
    x2, y2, p2 = pose1

    phi1 = arctan2(y2 - y1, x2 - x1) - p1
    d = sqrt((y2 - y1)**2 + (x2 - x1)**2)
    phi2 = p2 - p1 - phi1

    return phi1, d, phi2


def motion_decompose_demo2_plot(axes, x0=0, y0=0, theta0=0, v=4, omega=60):

    theta0 = radians(theta0)
    omega = radians(omega)

    dt = 1.0
    pose0 = (x0, y0, theta0)
    pose1 = motion_model(pose0, v, omega, dt)

    phi1, d, phi2 = odom_decompose(pose1, pose0)

    axes.clear()
    Pose(x0, y0, theta0).draw_axes(axes)

    Pose(*pose1).draw_axes(axes)

    robot_draw(axes, pose0[0], pose0[1], pose0[2], colour='blue')
    robot_draw(axes, pose1[0], pose1[1], pose1[2], colour='orange')

    if omega != 0:
        r = abs(v / omega)

        if omega < 0:
            theta0 += pi

        xc = x0 - r * sin(theta0)
        yc = y0 + r * cos(theta0)

        theta1 = 0
        theta2 = 2 * pi

        arc = Arc((xc, yc), 2 * r, 2 * r, angle=0, theta1=degrees(theta2),
                  theta2=degrees(theta1), linestyle='dashed', color='blue')
        axes.add_patch(arc)

    axes.plot((pose0[0], pose1[0]), (pose0[1], pose1[1]), color='black')

    dx = pose1[0] - pose0[0]
    dy = pose1[1] - pose0[1]
    axes.plot((pose1[0], pose1[0] + 0.25 * dx),
              (pose1[1], pose1[1] + 0.25 * dy),
              linestyle='dashed', color='black')

    axes.axis('equal')
    # axes.set_xlim(-2.5, 2.5)
    # axes.set_ylim(-2.5, 2.5)
    axes.set_xlim(-1, 4)
    axes.set_ylim(-1, 4)
    axes.grid(True)

    axes.set_title(r'$\phi_1 = %.1f$ deg, $d = %.1f$ m, $\phi_2 = %.1f$ deg' %
                   (degrees(phi1), d, degrees(phi2)))


def motion_decompose_demo2():

    fig, axes = subplots(figsize=(10, 5))
    show()

    interact(motion_decompose_demo2_plot, axes=fixed(axes),
             x0=(-2, 2, 0.1), y0=(-2, 2, 0.1), theta0=(-180, 180, 15),
             v=(0, 4, 0.2), omega=(-60, 60, 15),
             continuous_update=False)
