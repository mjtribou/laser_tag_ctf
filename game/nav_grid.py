# game/nav_grid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import heapq
import math

# World is Panda3D: X right, Y forward, Z up.
# We build a 2D occupancy grid in X–Y, inflating blocks by agent radius.

@dataclass
class Cell:
    ix: int
    iy: int

@dataclass
class NavGrid:
    origin_x: float
    origin_y: float
    cell: float
    nx: int
    ny: int
    occ: List[List[bool]]  # True = blocked

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c.ix < self.nx and 0 <= c.iy < self.ny

    def passable(self, c: Cell) -> bool:
        return not self.occ[c.iy][c.ix]

    def neighbors8(self, c: Cell) -> List[Tuple[Cell, float]]:
        res = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nc = Cell(c.ix + dx, c.iy + dy)
                if not self.in_bounds(nc):
                    continue
                if not self.passable(nc):
                    continue
                # Corner cutting guard for diagonals
                if dx != 0 and dy != 0:
                    a = Cell(c.ix + dx, c.iy)
                    b = Cell(c.ix, c.iy + dy)
                    if not (self.in_bounds(a) and self.in_bounds(b) and self.passable(a) and self.passable(b)):
                        continue
                cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
                res.append((nc, cost))
        return res

    def world_to_cell(self, x: float, y: float) -> Cell:
        ix = int(math.floor((x - self.origin_x) / self.cell))
        iy = int(math.floor((y - self.origin_y) / self.cell))
        return Cell(ix, iy)

    def cell_center(self, c: Cell) -> Tuple[float, float]:
        return (self.origin_x + (c.ix + 0.5) * self.cell,
                self.origin_y + (c.iy + 0.5) * self.cell)


def nearest_passable_xy(nav: NavGrid, x: float, y: float, max_radius: int = 8) -> Tuple[float, float]:
    """
    Returns the center of the nearest passable cell to (x,y).
    Searches outward in Manhattan rings up to max_radius cells.
    Falls back to clamped cell center if none found.
    """
    c0 = nav.world_to_cell(x, y)
    if nav.in_bounds(c0) and nav.passable(c0):
        return nav.cell_center(c0)

    # Expand in rings: |dx| + |dy| == r
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            dy = r - abs(dx)
            for sy in (-1, 1) if dy != 0 else (1,):
                nc = Cell(c0.ix + dx, c0.iy + sy * dy)
                if nav.in_bounds(nc) and nav.passable(nc):
                    return nav.cell_center(nc)

    # Clamp to grid if truly boxed in
    ix = max(0, min(nav.nx - 1, c0.ix))
    iy = max(0, min(nav.ny - 1, c0.iy))
    return nav.cell_center(Cell(ix, iy))

def _point_inside_inflated_block(x: float, y: float, z: float,
                                 center: Tuple[float, float, float],
                                 size: Tuple[float, float, float],
                                 inflate_xy: float) -> bool:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5 + inflate_xy, sy * 0.5 + inflate_xy, sz * 0.5
    # Only consider blocks that span the camera/player height
    if not (cz - hz <= z <= cz + hz):
        return False
    return (abs(x - cx) <= hx) and (abs(y - cy) <= hy)


def build_navgrid(mapdata, cell: float = 1.0, agent_radius: float = 0.38) -> NavGrid:
    # mapdata.bounds = (size_x, size_y_as_world_Y)
    size_x, size_y = mapdata.bounds
    origin_x = -size_x * 0.5
    origin_y = -size_y * 0.5
    nx = max(1, int(math.ceil(size_x / cell)))
    ny = max(1, int(math.ceil(size_y / cell)))
    occ = [[False for _ in range(nx)] for _ in range(ny)]

    # Mark blocked cells by rasterizing each block's inflated XY footprint.
    # Only blocks spanning the player/camera height contribute (sample_z).
    sample_z = 0.9
    inflate = float(agent_radius)

    def to_ix(x: float) -> int:
        return int(math.floor((x - origin_x) / cell))

    def to_iy(y: float) -> int:
        return int(math.floor((y - origin_y) / cell))

    for b in getattr(mapdata, "blocks", []):
        cx, cy, cz = b.pos
        sx, sy, sz = b.size
        hz = 0.5 * sz
        if not (cz - hz <= sample_z <= cz + hz):
            continue  # this stack of cubes does not intersect the play height

        hx = 0.5 * sx + inflate
        hy = 0.5 * sy + inflate

        ix0 = max(0, min(nx - 1, to_ix(cx - hx)))
        ix1 = max(0, min(nx - 1, to_ix(cx + hx)))
        iy0 = max(0, min(ny - 1, to_iy(cy - hy)))
        iy1 = max(0, min(ny - 1, to_iy(cy + hy)))

        for iy in range(iy0, iy1 + 1):
            row = occ[iy]
            for ix in range(ix0, ix1 + 1):
                row[ix] = True

    # NEW: add a perimeter ring of blocked cells to prevent paths that "lean" on world edges
    ring = max(1, int(math.ceil(agent_radius / cell)))
    # Left & right vertical strips
    for iy in range(ny):
        for k in range(ring):
            occ[iy][k] = True                    # left
            occ[iy][nx - 1 - k] = True           # right
    # Top & bottom horizontal strips
    for ix in range(nx):
        for k in range(ring):
            occ[k][ix] = True                    # bottom
            occ[ny - 1 - k][ix] = True           # top

    return NavGrid(nx=nx, ny=ny, cell=cell, origin_x=origin_x, origin_y=origin_y, occ=occ)

INF = 10**12

def dijkstra_field(nav: NavGrid, goals_world: List[Tuple[float, float]]):
    """
    Multi-source Dijkstra from 'goals_world' (world XY) across the grid.
    Returns (dist, parent) where:
      - dist[iy][ix] is cost-to-goal (float)
      - parent[iy][ix] is (px,py) cell you should step to NEXT to get closer (or (-1,-1) at sources)
    """
    nx, ny = nav.nx, nav.ny
    dist   = [[INF for _ in range(nx)] for _ in range(ny)]
    parent = [[(-1, -1) for _ in range(nx)] for _ in range(ny)]
    pq: List[Tuple[float, int, int]] = []  # (d, ix, iy)

    # seed with all goal cells that are passable
    for gx, gy in goals_world:
        c = nav.world_to_cell(gx, gy)
        if nav.in_bounds(c) and nav.passable(c):
            if dist[c.iy][c.ix] > 0.0:
                dist[c.iy][c.ix] = 0.0
                heapq.heappush(pq, (0.0, c.ix, c.iy))
                parent[c.iy][c.ix] = (-1, -1)

    while pq:
        d, ix, iy = heapq.heappop(pq)
        if d != dist[iy][ix]:
            continue  # stale
        cur = Cell(ix, iy)
        for nxt, step_cost in nav.neighbors8(cur):
            nd = d + step_cost
            if nd < dist[nxt.iy][nxt.ix]:
                dist[nxt.iy][nxt.ix] = nd
                parent[nxt.iy][nxt.ix] = (ix, iy)   # step from nxt -> parent to approach goal
                heapq.heappush(pq, (nd, nxt.ix, nxt.iy))

    return dist, parent

def next_step_toward(nav: NavGrid, parent, x: float, y: float) -> Tuple[float, float]:
    """
    From world (x,y), pick the next cell center following the precomputed parent field.
    If you're in an unpassable cell (rare), hop to nearest passable first.
    """
    c = nav.world_to_cell(x, y)
    if not (nav.in_bounds(c) and nav.passable(c)):
        x, y = nearest_passable_xy(nav, x, y, max_radius=8)
        c = nav.world_to_cell(x, y)

    px, py = parent[c.iy][c.ix]
    # At the target (parent=-1,-1) or unknown → stay in cell
    if px < 0 or py < 0:
        return nav.cell_center(c)
    return nav.cell_center(Cell(px, py))

def astar(nav: NavGrid, start_xy: Tuple[float, float], goal_xy: Tuple[float, float],
          max_iter: int = 20000) -> List[Tuple[float, float]]:
    start = nav.world_to_cell(*start_xy)
    goal  = nav.world_to_cell(*goal_xy)

    if not (nav.in_bounds(start) and nav.in_bounds(goal) and nav.passable(start) and nav.passable(goal)):
        return []

    def h(a: Cell, b: Cell) -> float:
        dx = a.ix - b.ix
        dy = a.iy - b.iy
        return math.hypot(dx, dy)

    openq: List[Tuple[float, int, Cell]] = []
    heapq.heappush(openq, (0.0, 0, start))
    came_from: Dict[Tuple[int,int], Optional[Cell]] = {(start.ix, start.iy): None}
    gscore: Dict[Tuple[int,int], float] = {(start.ix, start.iy): 0.0}

    it = 0
    while openq and it < max_iter:
        _, _, cur = heapq.heappop(openq)
        it += 1
        if cur.ix == goal.ix and cur.iy == goal.iy:
            # Reconstruct
            path_cells: List[Cell] = []
            k = (cur.ix, cur.iy)
            while k is not None:
                c = Cell(k[0], k[1])
                path_cells.append(c)
                prev = came_from[k]
                k = (prev.ix, prev.iy) if prev is not None else None
            path_cells.reverse()
            return [nav.cell_center(c) for c in path_cells]

        for nxt, step_cost in nav.neighbors8(cur):
            nk = (nxt.ix, nxt.iy)
            cand = gscore[(cur.ix, cur.iy)] + step_cost
            if cand < gscore.get(nk, float("inf")):
                gscore[nk] = cand
                prio = cand + h(nxt, goal)
                came_from[nk] = cur
                heapq.heappush(openq, (prio, it, nxt))

    return []

def astar_bounded(nav: NavGrid,
                  start_xy: Tuple[float, float],
                  goal_xy: Tuple[float, float],
                  w: float = 1.3,
                  pad: int = 10,
                  max_iter: int = 20000) -> List[Tuple[float, float]]:
    """
    Fast A* fallback:
      - weighted: f = g + w*h  (w~1.3 trims expansions)
      - bounded to an AABB around start↔goal, expanded by 'pad' cells
      - arrayified: flat lists, int parents; no dict churn
    """
    s = nav.world_to_cell(*start_xy)
    g = nav.world_to_cell(*goal_xy)
    if not (nav.in_bounds(s) and nav.in_bounds(g) and nav.passable(s) and nav.passable(g)):
        return []

    ix0 = max(0, min(s.ix, g.ix) - pad)
    iy0 = max(0, min(s.iy, g.iy) - pad)
    ix1 = min(nav.nx - 1, max(s.ix, g.ix) + pad)
    iy1 = min(nav.ny - 1, max(s.iy, g.iy) + pad)
    W, H = ix1 - ix0 + 1, iy1 - iy0 + 1
    N = W * H

    def to_idx(ix: int, iy: int) -> int: return (iy - iy0) * W + (ix - ix0)
    def from_idx(k: int) -> Tuple[int, int]: return (ix0 + (k % W), iy0 + (k // W))

    s_i = to_idx(s.ix, s.iy)
    g_i = to_idx(g.ix, g.iy)

    G = [float("inf")] * N
    P = [-1] * N
    CLOSED = [False] * N

    def h_ix(ix: int, iy: int) -> float:
        return math.hypot(ix - g.ix, iy - g.iy)

    G[s_i] = 0.0
    openq: List[Tuple[float, int, int]] = []  # (f, tie, idx)
    tie = 0
    heapq.heappush(openq, (w * h_ix(s.ix, s.iy), tie, s_i))

    it = 0
    while openq and it < max_iter:
        _, _, cur_i = heapq.heappop(openq)
        if CLOSED[cur_i]:
            continue
        CLOSED[cur_i] = True
        it += 1
        if cur_i == g_i:
            # reconstruct
            out: List[Tuple[float, float]] = []
            k = cur_i
            while k != -1:
                ix, iy = from_idx(k)
                out.append(nav.cell_center(Cell(ix, iy)))
                k = P[k]
            out.reverse()
            return out

        cur_ix, cur_iy = from_idx(cur_i)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nix, niy = cur_ix + dx, cur_iy + dy
                if not (ix0 <= nix <= ix1 and iy0 <= niy <= iy1):
                    continue
                if not nav.passable(Cell(nix, niy)):
                    continue
                # corner-cut guard
                if dx != 0 and dy != 0:
                    a = Cell(cur_ix + dx, cur_iy)
                    b = Cell(cur_ix, cur_iy + dy)
                    if not (nav.in_bounds(a) and nav.in_bounds(b) and nav.passable(a) and nav.passable(b)):
                        continue

                step = math.sqrt(2.0) if dx and dy else 1.0
                ni = to_idx(nix, niy)
                if CLOSED[ni]:
                    continue
                cand = G[cur_i] + step
                if cand < G[ni]:
                    G[ni] = cand
                    P[ni] = cur_i
                    tie += 1
                    f = cand + w * h_ix(nix, niy)
                    heapq.heappush(openq, (f, tie, ni))

    return []
