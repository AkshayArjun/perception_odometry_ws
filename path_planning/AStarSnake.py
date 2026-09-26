import numpy as np
from plotly import graph_objects as go
from dataclasses import dataclass, field
import math
import heapq

@dataclass
class Cylinder:
    center : np.ndarray
    radius : float
    known : bool

    def contains(self, point, margin=0.0):
        return self.radius + margin >= np.linalg.norm(point - self.center)

    def plot_2d(self, n=64):
        # filled circle; known = solid red, unknown = faded grey dashed
        theta = np.linspace(0, 2 * np.pi, n)
        xs = self.center[0] + self.radius * np.cos(theta)
        ys = self.center[1] + self.radius * np.sin(theta)
        color = "firebrick" if self.known else "gray"
        return go.Scatter(
            x=xs, y=ys,
            mode="lines", fill="toself",
            line=dict(color=color, dash="solid" if self.known else "dash"),
            opacity=0.8 if self.known else 0.4,
            showlegend=False,
            hoverinfo="skip",
        )


@dataclass
class World:
    min_bounds : np.ndarray
    max_bounds : np.ndarray
    num_obs : int
    seed : int = 0
    min_radius : float = 0.1
    max_radius : float = 0.5
    min_gap : float | None = 0.0     # None → overlaps allowed; float → min clearance between circle edges
    max_attempts : int = 1000        # stop sampling if circles won't fit
    obstacles : list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def obstacle_factory(self):
        # rejection sampling: draw one candidate at a time, keep it only if it clears existing obstacles
        attempts = 0
        while len(self.obstacles) < self.num_obs and attempts < self.max_attempts:
            attempts += 1
            point = self.rng.uniform(self.min_bounds, self.max_bounds)
            radius = self.rng.uniform(self.min_radius, self.max_radius)
            if self.min_gap is not None and any(
                np.linalg.norm(point - obs.center) < radius + obs.radius + self.min_gap
                for obs in self.obstacles
            ):
                continue
            is_known = bool(self.rng.choice([True, False], p=[0.65, 0.35]))  # 65% known, 35% unknown
            self.obstacles.append(Cylinder(point, radius, is_known))

        if len(self.obstacles) < self.num_obs:
            print(f"warning: placed {len(self.obstacles)}/{self.num_obs} obstacles "
                  f"after {self.max_attempts} attempts (min_gap={self.min_gap})")

    def is_valid(self, point, margin, known_only=False):
        if np.any(point < self.min_bounds) or np.any(point > self.max_bounds):
            return False
        for obs in self.obstacles:
            if known_only and not obs.known:
                continue
            if obs.contains(point, margin):
                return False
        return True

    def plot_2d(self):
        fig = go.Figure([obs.plot_2d() for obs in self.obstacles])
        fig.update_layout(
            xaxis=dict(range=[self.min_bounds[0], self.max_bounds[0]]),
            yaxis=dict(range=[self.min_bounds[1], self.max_bounds[1]],
                       scaleanchor="x", scaleratio=1),   # equal aspect so circles look round
            width=600, height=600,
            template="plotly_white",
        )
        return fig


class Grid:
    # 8-connected steps; a step is diagonal when both components are nonzero → cost √2, else 1 (costs in cells)
    OFFSETS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],
                        [-1, -1], [-1, 1], [1, -1], [1, 1]])
    COSTS = np.where(np.all(np.abs(OFFSETS) == 1, axis=1), np.sqrt(2), 1.0)

    def __init__(self, world, resolution, margin=0.0):
        self.world = world
        self.resolution = resolution
        self.margin = margin
        # every point in a cell is within res·√2/2 of its center, so inflating obstacles by
        # that much marks every cell an obstacle touches as occupied (no obstacle can slip between centers)
        self.inflation = margin + resolution * np.sqrt(2) / 2

        self.shape = np.ceil((world.max_bounds - world.min_bounds) / resolution).astype(int)
        self.origin = world.min_bounds

    def world_to_grid(self, points): 
        return np.floor((np.asarray(points) - self.origin)/self.resolution).astype(int)

    def grid_to_world (self, indices):
        return self.origin + (np.asarray(indices) + 0.5) * self.resolution

    def check_grid_occupancy(self, known_only=False):
        occupancy = np.zeros(self.shape, dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                point = self.grid_to_world(np.array([i, j]))
                occupancy[i, j] = not self.world.is_valid(point, self.inflation, known_only)
        return occupancy

    def is_valid(self, indices, occupancy):
        i,j = indices
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            return False
        return not occupancy[i, j]

    def neighbors(self, indices, occupancy, corner_cutting=False):    
        indices = np.asarray(indices)
        result = []
        for offset, cost in zip(self.OFFSETS, self.COSTS):
            n = indices + offset
            if not self.is_valid(n, occupancy):
                continue
            if cost > 1 and not corner_cutting and not (
                self.is_valid(indices + [offset[0], 0], occupancy)
                and self.is_valid(indices + [0, offset[1]], occupancy)
            ):
                continue
            result.append((tuple(int(v) for v in n), float(cost)))
        return result

    def plot_2d(self, occupancy, title=""):
        # occupied cells as a heatmap in world coords, with the true obstacle outlines drawn on top
        xs = self.origin[0] + (np.arange(self.shape[0]) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(self.shape[1]) + 0.5) * self.resolution
        ii, jj = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        cell_ids = np.char.add(np.char.add("(", ii.astype(str)), np.char.add(", ", np.char.add(jj.astype(str), ")")))
        fig = go.Figure(go.Heatmap(
            x=xs, y=ys, z=occupancy.T.astype(int),      # .T: heatmap rows are y, our first index is x
            text=cell_ids.T,
            colorscale=[[0, "white"], [1, "lightsteelblue"]], zmin=0, zmax=1,
            showscale=False, xgap=1, ygap=1,
            hovertemplate="cell %{text}<br>x=%{x:.2f}, y=%{y:.2f}<br>occupied=%{z}<extra></extra>",
        ))
        for obs in self.world.obstacles:
            fig.add_trace(obs.plot_2d().update(fill="none", opacity=1))   # outline only so cells stay visible
        fig.update_layout(
            title=title,
            xaxis=dict(range=[self.world.min_bounds[0], self.world.max_bounds[0]]),
            yaxis=dict(range=[self.world.min_bounds[1], self.world.max_bounds[1]],
                       scaleanchor="x", scaleratio=1),
            width=600, height=600,
            template="plotly_white",
            plot_bgcolor="lightgray",   # shows through the xgap/ygap as grid lines
        )
        return fig

class AStar():
    def __init__(self, grid, occupancy):
        self.grid = grid
        self.occupancy = occupancy

    def heuristic(self, a, b): 
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        min_h = min(dx, dy)
        max_h = max(dx, dy)
        return math.sqrt(2)*min_h + (max_h - min_h)

    def search(self, start, goal, corner_cutting=False):
        # returns list of cells start → goal, or None if unreachable; cost / expansions saved for inspection
        start = tuple(int(v) for v in start)
        goal = tuple(int(v) for v in goal)
        self.cost, self.expanded = None, 0
        if not self.grid.is_valid(start, self.occupancy) or not self.grid.is_valid(goal, self.occupancy):
            return None

        # per-cell bookkeeping stored as grids (GfG's closed_list / cell_details), indexed by cell tuple;
        # created fresh each search so repeated calls (replanning) never see stale state
        g = np.full(self.grid.shape, np.inf)
        parent = np.full((*self.grid.shape, 2), -1, dtype=int)
        closed = np.zeros(self.grid.shape, dtype=bool)

        g[start] = 0.0
        h = self.heuristic(start, goal)
        open_list = [(h, h, start)]   # (f, h, cell): ties on f go to the cell closer to the goal

        while open_list:
            _, _, cell = heapq.heappop(open_list)
            if closed[cell]:
                continue   # stale duplicate: a cheaper copy of this cell was already expanded
            if cell == goal:   # checked on pop, not push, so the path is guaranteed shortest
                self.cost = float(g[goal])
                return self.trace_path(parent, start, goal)
            closed[cell] = True
            self.expanded += 1

            for nbr, step in self.grid.neighbors(cell, self.occupancy, corner_cutting):
                if closed[nbr]:
                    continue
                g_new = g[cell] + step
                if g_new < g[nbr]:
                    g[nbr] = g_new
                    parent[nbr] = cell
                    h = self.heuristic(nbr, goal)
                    heapq.heappush(open_list, (g_new + h, h, nbr))

        return None

    def trace_path(self, parent, start, goal):
        path = [goal]
        while path[-1] != start:
            path.append(tuple(int(v) for v in parent[path[-1]]))
        path.reverse()
        return path
