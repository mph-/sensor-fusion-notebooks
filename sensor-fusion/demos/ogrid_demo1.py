# Michael P. Hayes UCECE, Copyright 2018--2026
from numpy import linspace, radians, ceil, cos, sin, round, arange, sqrt
from numpy import argwhere, zeros, exp, log
from ipywidgets import interact
from matplotlib.pyplot import subplots
from .lib.robot import Robot
from .lib.line import LineSeg

xmin = -10
xmax = 10
ymin = 0
ymax = 10
tmin = -180
tmax = 180

Nx = 21
Ny = 11

x = linspace(xmin, xmax, Nx)
y = linspace(ymin, ymax, Ny)

beamwidth = 15

class Scan(object):

    def __init__(self, angles, ranges, rmax):
        self.angles = angles
        self.ranges = ranges
        self.rmax = rmax


class Rangefinder(object):

    def __init__(self, beamwidth, rmax=20):
        self.beamwidth = beamwidth
        self.rmax = rmax

    def scan(self, pose, walls, dangle=radians(1)):

        xr, yr, hr = pose.x, pose.y, pose.theta

        Nrays = int(ceil(self.beamwidth / dangle))
        angles = linspace(hr - 0.5 * (self.beamwidth - dangle),
                          hr + 0.5 * (self.beamwidth - dangle), Nrays)

        ranges = angles * 0

        for m, angle in enumerate(angles):

            x = xr + cos(angle) * self.rmax
            y = yr + sin(angle) * self.rmax

            raylineseg = LineSeg((xr, yr), (x, y))

            rmin = self.rmax
            blockhit = None
            for wall in walls:
                R, r1, block = wall.intersection(raylineseg)
                if R and r1 < rmin:
                    rmin = r1
                    blockhit = block

            ranges[m] = rmin

        return Scan(angles, ranges, self.rmax)

    def draw_scan(self, axes, pose, scan, **kwargs):

        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()

        axes.set_xlim(xmin, xmax)
        axes.set_ylim(ymin, ymax)

        xr, yr = pose.x, pose.y

        dangle = radians(1)

        for angle, r in zip(scan.angles, scan.ranges):

            t1 = angle - 0.5 * dangle
            t2 = angle + 0.5 * dangle

            x1 = xr + cos(t1) * r
            x2 = xr + cos(t2) * r
            y1 = yr + sin(t1) * r
            y2 = yr + sin(t2) * r

            axes.fill((xr, x1, x2), (yr, y1, y2),
                      alpha=kwargs.pop('alpha', 0.5),
                      color=kwargs.pop('color', 'tab:blue'),
                      **kwargs)


class Occfind(object):

    def __init__(self, pose, scan, walls):
        self.pose = pose
        self.scan = scan
        self.walls = walls

    def hits(self):

        hits = {}

        xr, yr = self.pose.x, self.pose.y

        for angle, r in zip(self.scan.angles, self.scan.ranges):

            if r >= self.scan.rmax:
                continue

            xt = xr + r * cos(angle)
            yt = yr + r * sin(angle)

            raylineseg = LineSeg((xr, yr), (xt, yt))

            blockhit = None
            rmin = self.scan.rmax

            for wall in self.walls:
                R, r1, block = wall.intersection(raylineseg)
                if R and r1 < rmin:
                    rmin = r1
                    blockhit = block

            if blockhit is None:
                continue

            hit = blockhit.x, blockhit.y

            if hit not in hits:
                hits[hit] = 0
            hits[hit] += 1
        return hits

    def visits(self):

        dr = 0.1
        visits = {}

        for angle, r in zip(self.scan.angles, self.scan.ranges):

            xt = self.pose.x + r * cos(angle)
            yt = self.pose.y + r * sin(angle)

            lineseg = LineSeg((self.pose.x, self.pose.y), (xt, yt))

            length = lineseg.length
            steps = int(length / dr + 0.5)
            t = linspace(0, 1, steps)

            for t1 in t:
                xt, yt = lineseg.coord(t1)
                xc = int(round(xt))
                yc = int(round(yt))
                visit = (xc, yc)

                if visit not in visits:
                    visits[visit] = 0
                visits[visit] += 1
        return visits

    def hits_misses(self):

        hits = self.hits()
        visits = self.visits()

        misses = {}
        for visit in visits:
            if visit not in hits:
                misses[visit] = visits[visit]
        return hits, misses


class Block(object):

    def __init__(self, x, y, d=0.5):

        self.x, self.y = x, y

        pa = x - d, y - d
        pb = x - d, y + d
        pc = x + d, y + d
        pd = x + d, y - d

        self.linesegs = (LineSeg(pa, pb), LineSeg(pb, pc),
                         LineSeg(pc, pd), LineSeg(pd, pa))

    def __str__(self):
        return "%d, %d" % (self.x, self.y)


class Wall(object):

    def __init__(self, p0, p1, d=0.5):

        self.p0 = p0
        self.p1 = p1
        self.d = d

        x0, y0 = p0
        x1, y1 = p1

        self.blocks = []
        if x0 == x1:
            # Vertical
            for y in arange(y0, y1 + 1):
                self.blocks.append(Block(x0, y, d))
        elif y0 == y1:
            # Horizontal
            for x in arange(x0, x1 + 1):
                self.blocks.append(Block(x, y0, d))

    def draw(self, axes):

        x0, y0 = self.p0
        x1, y1 = self.p1

        d = self.d

        x = [x0 - d, x0 - d, x1 + d, x1 + d, x0 - d]
        y = [y0 - d, y1 + d, y1 + d, y0 - d, y0 - d]
        axes.plot(x, y, 'k')

    def intersection(self, ray):

        xr, yr = ray.p0
        rmin = 1000
        blockhit = None
        hit = False

        # Check all four walls and return closest hit if any
        for block in self.blocks:
            for lineseg in block.linesegs:
                R = ray.intersection(lineseg)
                if R:
                    xt, yt = R
                    r = sqrt((xt - xr)**2 + (yt - yr)**2)
                    if r < rmin:
                        rmin = r
                        hit = R
                        blockhit = block

        return hit, rmin, blockhit

wall1 = Wall((-1, 8), (3, 8))
wall2 = Wall((4, 6), (4, 7))
wall3 = Wall((-7, 2), (-7, 7))
walls = (wall1, wall2, wall3)


def heatmap(ax, x, y, data, fmt='%.1f', skip=[], **kwargs):

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # These are the corners.
    xc = linspace(x[0] - dx / 2, x[-1] + dx / 2, len(x) + 1)
    yc = linspace(y[0] - dy / 2, y[-1] + dy / 2, len(y) + 1)

    c = ax.pcolor(xc, yc, data, linewidths=4, vmin=0.0, vmax=1.0,
                  edgecolors=kwargs.pop('edgecolors', 'w'),
                  cmap=kwargs.pop('cmap', 'Purples'),
                  **kwargs)

    c.update_scalarmappable()

    for p, color, value in zip(c.get_paths(), c.get_facecolors(),
                               c.get_array().flatten()):
        x, y = p.vertices[:-2, :].mean(0)

        if all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)

        draw_text = True
        for skip1 in skip:
            xs, ys = skip1
            if abs(x - xs) < 0.1 and abs(y - ys) < 0.1:
                draw_text = False
                break

        if draw_text:
            # ?????
            ax.text(x, y, fmt % value, ha="center", va="center",
                    color=color, **kwargs)


class Ogrid(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        # Log odds
        self.grid = zeros((Ny, Nx))

    def draw(self, axes, skip=[]):
        # Convert log odds to probability
        P = 1 - 1 / (1 + exp(self.grid))
        heatmap(axes, self.x, self.y, P, skip=skip)

    def update(self, cells, P1, P2):

        lam = log(P1 / P2)
        for cell in cells:
            # Weed out marginal cells.
            # if cells[cell] < 5:
            #    continue

            try:
                m = argwhere(self.x == cell[0])[0][0]
                n = argwhere(self.y == cell[1])[0][0]
                self.grid[n, m] += lam
            except Exception:
                pass


ogrid = Ogrid(x, y)
rangefinder = Rangefinder(radians(beamwidth))


def ogrid_demo1_plot(x=3, y=1, heading=75):

    robot = Robot(x, y, heading=radians(heading))

    scan = rangefinder.scan(robot.pose, walls)
    hits, misses = Occfind(robot.pose, scan, walls).hits_misses()

    ogrid.update({(robot.x, robot.y): 100}, 0.001, 1)
    ogrid.update(hits, 0.06, 0.005)
    ogrid.update(misses, 0.2, 0.9)

    fig, ax = subplots(figsize=(10, 5))
    ax.axis('equal')
    ogrid.draw(ax, ((robot.x, robot.y), ))
    robot.draw(ax, d=0.55)
    for wall in walls:
        wall.draw(ax)
    rangefinder.draw_scan(ax, robot.pose, scan)


def ogrid_demo1():
    interact(ogrid_demo1_plot,
             x=(xmin, xmax, 1),
             y=(ymin, ymax, 1),
             heading=(tmin, tmax, 15),
             continuous_update=False)
