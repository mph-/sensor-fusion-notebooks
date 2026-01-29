# Michael P. Hayes UCECE, Copyright 2018--2021
from numpy import radians, exp, linspace, zeros, sqrt, nanargmax, unravel_index
from numpy import nan
from matplotlib.pyplot import Circle
from ipywidgets import interact
from matplotlib.pyplot import subplots
from .lib.utils import angle_difference
from .lib.robot import robot_draw, Robot


def braking_distance(v, a_max):

    return 0.5 * v**2 / abs(a_max)


def obstacle_distance(x, y, obs, d=0):

    dx = x - obs[0]
    dy = y - obs[1]
    # Radius of inflated obstacle
    r = obs[2] + d

    # Check if inside obstacle
    Rsq = dx**2 + dy**2
    R = sqrt(Rsq)
    if R < r:
        return 0

    return abs(R - r)


def objective(speed, speed_goal, heading, heading_goal):

    w1 = exp(-abs(speed - speed_goal) / 1.0)
    w2 = exp(-abs(angle_difference(heading, heading_goal)) / radians(30))
    return w1 * w2


def calc_objective(weights, heading, vv, ww, speed_goal, heading_goal, dt,
                   obstacles, x, y, d, a_max):
    """d is the diameter of the robot"""

    ww = radians(ww)
    heading = radians(heading)
    heading_goal = radians(heading_goal)

    for m, w in enumerate(ww):
        for n, v in enumerate(vv):
            robot = Robot(x, y, heading)
            robot.transition(v, w, dt)

            d_brake = braking_distance(v, a_max)

            d_obs_min = 1e6
            for obs in obstacles:
                d_obs = obstacle_distance(robot.x, robot.y, obs, d)
                if d_obs < d_obs_min:
                    d_obs_min = d_obs

            clearance = d_obs_min - d_brake

            if clearance < 0:
                weights[n, m] = nan
            else:
                weights[n, m] = objective(v, speed_goal,
                                          robot.heading, heading_goal)
                # Should determine clearance to nearest obstacle and
                # penalise fast speeds close to obstacles.
                # (1 - exp(-clearance / d))


def dwa_demo2_plot(x=0, y=1, heading=90, v=1, omega=0, speed_goal=1,
                   heading_goal=90, obstacles=False, inflate=False,
                   show_best=False):

    dt = 1
    steps = 1

    # Robot diameter
    d = 0.25

    # Max speeds and accelerations
    v_max = 4
    omega_max = 360
    a_max = 2
    alpha_max = 180

    w = omega
    w_max = omega_max

    v_min = -v_max
    w_min = -w_max

    v1_max = v + a_max * dt
    v1_min = v - a_max * dt
    w1_max = w + alpha_max * dt
    w1_min = w - alpha_max * dt

    v1_min = max(v_min, v1_min)
    v1_max = min(v_max, v1_max)
    w1_min = max(w_min, w1_min)
    w1_max = min(w_max, w1_max)

    if obstacles:
        obstacles = ((0.5, 3, 0.5, 'purple'), )
    else:
        obstacles = ()

    extra_v = v_max * 0.05
    extra_w = w_max * 0.05

    fig, axes = subplots(1, 2, figsize=(12, 6))
    xax, vax = axes

    vax.set_xlim(v_min - extra_v, v_max + extra_v)
    vax.set_ylim(w_min - extra_w, w_max + extra_w)

    # Region of all possible speeds
    vax.plot((v_min, v_min, v_max, v_max, v_min),
             (w_min, w_max, w_max, w_min, w_min), 'b-')

    # Region of all achievable speeds
    vax.plot((v1_min, v1_min, v1_max, v1_max, v1_min),
             (w1_min, w1_max, w1_max, w1_min, w1_min),
             '-', color='orange')

    vax.set_xlabel('$v$')
    vax.set_ylabel('$\omega$')

    Nv = 9
    Nw = 9

    weights = zeros((Nv, Nw))

    vv = linspace(v1_min, v1_max, Nv)
    ww = linspace(w1_min, w1_max, Nw)

    dr = 0
    if inflate:
        dr = d

    calc_objective(weights, heading, vv, ww, speed_goal, heading_goal,
                   dt, obstacles, x, y, dr, a_max)

    max_index = nanargmax(weights)
    v_index, w_index = unravel_index(max_index, weights.shape)
    v_best = vv[v_index]
    w_best = ww[w_index]

    vax.imshow(weights.T, origin='lower', interpolation=None, aspect='auto',
               extent=(v1_min, v1_max, w1_min, w1_max))
    vax.grid(True)
    vax.set_title('$v = %.1f, \omega = %.1f$' % (v_best, w_best))

    xax.set_xlabel('$x$')
    xax.set_ylabel('$y$')
    xax.set_xlim(-2, 2)
    xax.set_ylim(0, 4)
    xax.grid(True)

    for obs in obstacles:
        circle = Circle((obs[0], obs[1]), obs[2], color=obs[3], fill=True)
        xax.add_artist(circle)
        if inflate:
            circle = Circle((obs[0], obs[1]), obs[2] + d, color=obs[3],
                            fill=True, alpha=0.5)
            xax.add_artist(circle)

    xv = zeros(steps + 1)
    yv = zeros(steps + 1)
    thetav = zeros(steps + 1)

    robot = Robot(x=x, y=y, heading=radians(heading))

    if show_best:
        v = v_best
        omega = w_best
    xax.set_title('$v = %.1f, \omega = %.1f$' % (v, omega))

    for m in range(steps + 1):
        xv[m] = robot.x
        yv[m] = robot.y
        thetav[m] = robot.heading
        robot.transition(v, radians(omega), dt=dt)

    # Perhaps should draw path

    robot_draw(xax, xv[0], yv[0], thetav[0], d)
    for m in range(1, len(xv)):
        robot_draw(xax, xv[m], yv[m], thetav[m], d, linestyle='dashed')


def dwa_demo2():
    interact(dwa_demo2_plot, x=(-4, 4, 0.5), y=(0, 4, 0.5),
             dt=(0.1, 1, 0.1),
             v=(0, 2, 0.1), omega=(-60, 60, 15),
             v_max=(1, 5), omega_max=(90, 360, 15),
             a_max=(0.5, 2, 0.5), alpha_max=(30, 180, 15),
             heading=(0, 180, 15), heading_goal=(0, 180, 15),
             speed_goal=(0, 2, 0.1), steps=(0, 5, 1),
             continuous_update=False)
