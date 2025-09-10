# client.py
import sys, asyncio, json, time, math, argparse, threading, random
from typing import Dict, Any, List, Tuple
from bisect import bisect, bisect_right

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import Vec3, Point3, DirectionalLight, AmbientLight, LVector3f, KeyboardButton, WindowProperties
from panda3d.core import LColor, MouseButton, LineSegs, TextNode
from panda3d.core import TextNode, TransparencyAttrib, NodePath, LVecBase4f
from panda3d.core import CompassEffect, BillboardEffect, LColor

from common.net import send_json, read_json, lan_discovery_broadcast
from game.constants import TEAM_RED, TEAM_BLUE
from game.map_gen import generate

# ========== Cosmetics Manager ==========
class CosmeticsManager:
    """
    Renders:
      - Headgear (ballcap, top_hat, headband) per player, color from palette (excludes blue/orange).
      - Nameplates (3D billboard text) for all or team-only; distance fade; toggle via config.
      - Teammate caret marker always-on-top and visible through walls.

    Usage:
      self.cosmetics = CosmeticsManager(self)
      self.cosmetics.attach(pid, player_np, name, team)
      self.cosmetics.update()
      self.cosmetics.detach(pid)
    """
    def __init__(self, client):
        self.client = client
        self.cfg = client.cfg.get("cosmetics", {})
        self.by_pid = {}  # pid -> dict(root, head_anchor, headgear_np, name_np, caret_np, team)
        # Cache palette and helpers
        hg = self.cfg.get("headgear", {})
        self.palette = {k: LVecBase4f(*v) for k, v in hg.get("palette", {}).items()}
        self.disallow = set(hg.get("disallow_colors", []))
        self.assignments = hg.get("assignments", {})
        self.head_z = float(self.cfg.get("anchors", {}).get("head_z", 1.15))

        # Try to load your unit cube model once; fall back to CardMaker if missing
        self.box = None
        try:
            # Adjust path if your asset lives elsewhere (e.g., "assets/box.egg")
            self.box = self.client.loader.loadModel("models/box.egg")
        except Exception:
            self.box = None  # we'll synthesize simple quads if needed

    # ---------- Public API ----------
    def attach(self, pid, player_np, name, team):
        """Create cosmetic nodes for a player."""
        if pid in self.by_pid:
            self.detach(pid)

        root = player_np.attachNewNode(f"cosmetics-{pid}")
        head_anchor = root.attachNewNode("head_anchor")
        head_anchor.setZ(self.head_z)

        # Headgear
        headgear_np = None
        if self.cfg.get("headgear", {}).get("enabled", True):
            headgear_np = head_anchor.attachNewNode("headgear")
            htype, color = self._resolve_headgear(pid, name)
            self._build_headgear(headgear_np, htype, color)

        # Nameplate
        name_np = None
        if self.cfg.get("nameplates", {}).get("enabled", True):
            show_for = self.cfg.get("nameplates", {}).get("for", "all")
            if show_for == "all" or (show_for == "team_only" and self._is_teammate(team)):
                name_np = self._make_nameplate(head_anchor, name)

        # Teammate caret (always-on-top)
        caret_np = None
        tm_cfg = self.cfg.get("teammate_marker", {})
        if tm_cfg.get("enabled", True) and self._is_teammate(team):
            caret_np = self._make_caret(head_anchor, tm_cfg.get("style", "caret"))

        self.by_pid[pid] = {
            "root": root,
            "head_anchor": head_anchor,
            "headgear_np": headgear_np,
            "name_np": name_np,
            "caret_np": caret_np,
            "team": team,
            "name": name,
        }

    def detach(self, pid):
        d = self.by_pid.pop(pid, None)
        if not d: return
        for k in ("caret_np", "name_np", "headgear_np", "root"):
            np = d.get(k)
            if np is not None:
                np.removeNode()

    def update(self):
        """Call every frame to do distance-based fading/visibility."""
        if not self.by_pid:
            return
        cam = self.client.camera  # ShowBase camera
        render = self.client.render
        # Settings
        np_cfg = self.cfg.get("nameplates", {})
        np_maxd = float(np_cfg.get("max_distance", 60.0))
        np_fade = float(np_cfg.get("fade_start", 40.0))

        tm_cfg = self.cfg.get("teammate_marker", {})
        tm_maxd = float(tm_cfg.get("max_distance", 60.0))
        tm_fade = float(tm_cfg.get("fade_start", 40.0))
        tm_alpha = float(tm_cfg.get("alpha", 0.85))

        for pid, d in self.by_pid.items():
            head_anchor = d["head_anchor"]
            head_world = head_anchor.getPos(render)
            cam_pos = cam.getPos(render)
            dist = (head_world - cam_pos).length()

            # Nameplate fade
            name_np = d.get("name_np")
            if name_np is not None:
                self._apply_fade(name_np, dist, np_fade, np_maxd, 1.0)

            # Caret fade (with base alpha)
            caret_np = d.get("caret_np")
            if caret_np is not None:
                self._apply_fade(caret_np, dist, tm_fade, tm_maxd, tm_alpha)

    # ---------- Helpers ----------
    def _is_teammate(self, team):
        try:
            return team == self.client.local_team
        except Exception:
            # Fallback if local_team not set yet
            return False

    def _resolve_headgear(self, pid, name):
        hg = self.cfg.get("headgear", {})
        default = hg.get("default", {"type": "ballcap", "color": "purple"})
        # Name mapping first, then pid mapping, then default/random
        spec = self.assignments.get(str(name)) or self.assignments.get(str(pid)) or default

        htype = spec.get("type", "ballcap")
        color_key = spec.get("color")
        color = self._pick_color(color_key)
        return htype, color

    def _pick_color(self, preference):
        # honor preference if valid and not disallowed
        if preference and (preference in self.palette) and (preference not in self.disallow):
            return self.palette[preference]
        # Otherwise pick a random allowed key
        keys = [k for k in self.palette.keys() if k not in self.disallow]
        if not keys:
            return LVecBase4f(1, 1, 1, 1)
        return self.palette[random.choice(keys)]

    def _apply_overlay_render_state(self, np, alpha=1.0, on_top=True):
        np.setTransparency(TransparencyAttrib.M_alpha)
        if on_top:
            np.setDepthTest(False)
            np.setDepthWrite(False)
            np.setBin("fixed", 0)
        if alpha is not None:
            np.setColorScale(1, 1, 1, alpha)

    def _apply_fade(self, np, dist, fade_start, max_dist, base_alpha):
        if dist >= max_dist:
            np.hide()
            return
        np.show()
        if dist <= fade_start:
            np.setColorScale(1, 1, 1, base_alpha)
            return
        # Linear fade to 0
        t = (dist - fade_start) / max(0.0001, (max_dist - fade_start))
        a = max(0.0, min(1.0, 1.0 - t)) * base_alpha
        np.setColorScale(1, 1, 1, a)

    # ---------- Builders ----------
    def _make_nameplate(self, parent, name):
        cfg = self.cfg.get("nameplates", {})
        scale = float(cfg.get("scale", 0.6))
        tn = TextNode(f"name-{name}")
        tn.setText(name)
        tn.setAlign(TextNode.ACenter)
        tn.setShadow(0.03, 0.03)
        tn.setShadowColor(0, 0, 0, 1)
        tn.setTextColor(1, 1, 1, 1)
        np = parent.attachNewNode(tn)
        np.setBillboardPointEye()
        np.setScale(scale)
        np.setZ(0.18)  # float it a bit above the headgear
        self._apply_overlay_render_state(np, alpha=1.0, on_top=True)  # render through walls
        return np

    def _make_caret(self, parent, style="caret"):
        # Simple caret using text; reliable and cheap
        tn = TextNode("caret")
        tn.setText("^" if style == "caret" else "ˇ")
        tn.setAlign(TextNode.ACenter)
        tn.setShadow(0.02, 0.02)
        tn.setShadowColor(0, 0, 0, 1)
        tn.setTextColor(1, 1, 1, 1)
        np = parent.attachNewNode(tn)
        np.setBillboardPointEye()
        scale = float(self.cfg.get("teammate_marker", {}).get("scale", 0.5))
        np.setScale(scale)
        np.setZ(0.34)
        self._apply_overlay_render_state(np, alpha=self.cfg.get("teammate_marker", {}).get("alpha", 0.85), on_top=True)
        return np

    def _build_headgear(self, parent, htype, color):
        # Build from scaled copies of a unit box
        maker = self._copy_box

        if htype == "top_hat":
            brim = maker(parent, scale=(1.15, 1.15, 0.05), pos=(0, 0, 0.03), color=color)
            crown = maker(parent, scale=(0.70, 0.70, 0.70), pos=(0, 0, 0.45), color=color)
            # Slight edge darkening via color scale on crown for definition
            crown.setColorScale(0.9*color.x, 0.9*color.y, 0.9*color.z, color.w)

        elif htype == "headband":
            z = -0.02
            t = 0.06  # thickness
            w = 0.75  # outside width
            d = 0.05  # depth off head
            # 4 thin slats forming a ring
            maker(parent, scale=(w, t, 0.10), pos=(0,  0.45, z), color=color)  # front
            maker(parent, scale=(w, t, 0.10), pos=(0, -0.45, z), color=color)  # back
            maker(parent, scale=(t, w, 0.10), pos=( 0.45, 0,   z), color=color)  # right
            maker(parent, scale=(t, w, 0.10), pos=(-0.45, 0,   z), color=color)  # left

        else:  # "ballcap" (default)
            crown = maker(parent, scale=(0.72, 0.72, 0.33), pos=(0, 0, 0.12), color=color)
            brim  = maker(parent, scale=(0.72, 1.05, 0.05), pos=(0, 0.43, 0.02), color=color)

    def _copy_box(self, parent, scale=(1,1,1), pos=(0,0,0), color=LVecBase4f(1,1,1,1)):
        if self.box is not None:
            np = self.box.copyTo(parent)
        else:
            # Minimal fallback: a tiny square “card” stack approximating a box
            from panda3d.core import CardMaker
            cm = CardMaker("quad")
            cm.setFrame(-0.5, 0.5, -0.5, 0.5)
            face = parent.attachNewNode(cm.generate())
            # Give it some thickness illusion by duplicating slightly
            face2 = face.copyTo(parent); face2.setY(0.999)
            np = parent.attachNewNode("fallback_box")
            face.reparentTo(np); face2.reparentTo(np)
        np.setScale(*scale)
        np.setPos(*pos)
        np.setTransparency(TransparencyAttrib.M_alpha)
        np.setColor(color)
        return np

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


class AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coro(self, coro):
        """Schedule a coroutine onto the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class NetworkClient:
    def __init__(self, cfg, name="Player"):
        self.cfg = cfg
        self.name = name
        self.reader = None
        self.writer = None
        self.pid = None
        self.team = None
        self.state = None
        self.last_input = {}

    async def connect(self, host: str, port: int):
        self.reader, self.writer = await asyncio.open_connection(host, port)
        await send_json(self.writer, {"type": "hello", "name": self.name})
        welcome = await read_json(self.reader)
        self.pid = welcome["pid"]
        self.team = welcome["team"]
        print(f"[net] connected. pid={self.pid} team={self.team}")

    async def recv_state_loop(self, on_state):
        if self.reader is None:
            raise RuntimeError("recv_state_loop called before connect() completed")
        while True:
            msg = await read_json(self.reader)
            if not msg:
                break
            if msg.get("type") == "state":
                self.state = msg
                on_state(msg)

    async def send_input(self, data: Dict[str, Any]):
        self.last_input = data
        await send_json(self.writer, {"type": "input", "data": data})


def _angle_lerp_deg(a: float, b: float, t: float) -> float:
    """
    Shortest-path spherical (mod 360) interpolation for degrees.
    """
    # Map delta into [-180, 180]
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + d * t


class GameApp(ShowBase):
    def __init__(self, cfg, host: str, port: int, name: str, interp_delay: float, interp_predict: float = 0.0):
        ShowBase.__init__(self)
        self.set_background_color(0.05, 0.05, 0.07, 1)
        self.disableMouse()

        cam_cfg = cfg.get("camera", {})
        near = float(cam_cfg.get("near", 0.03))
        far  = float(cam_cfg.get("far", 300.0))
        base.camLens.setNear(near)
        base.camLens.setFar(far)

        self.cfg = cfg
        self.client = NetworkClient(cfg, name=name)
        self.host, self.port = host, port
        self.yaw, self.pitch = 0.0, 0.0

        # --- Interpolation state ---
        # We keep server snapshots sorted by their embedded server time.
        self.snapshots: List[Dict[str, Any]] = []
        self.snap_times: List[float] = []
        self.latest_server_time: float = 0.0
        self.interp_delay: float = max(0.0, float(interp_delay))
        self.interp_predict: float = max(0.0, float(interp_predict))
        print(f"[init] Interp delay = {int(self.interp_delay * 1000)} ms, predict = {int(self.interp_predict*1000)} ms")
        self.render_time: float = 0.0

        # lighting
        dlight = DirectionalLight("dlight")
        dlight.setColor(LColor(0.9, 0.9, 0.9, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -60, 0)
        self.render.setLight(dlnp)

        alight = AmbientLight("alight")
        alight.setColor(LColor(0.2, 0.2, 0.25, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # world (visual & simple bullet for occlusion later)
        # (Bullet not yet used here; scaffold kept for future)
        # self.world = BulletWorld(); self.world.setGravity(Vec3(0,0,-9.81))

        # arena
        size_x, size_y = cfg["gameplay"]["arena_size_m"]
        self.mapdata = generate(seed=42, size_x=size_x, size_z=size_y)

        # floor plane (visual only)
        cm = self.render.attachNewNode("floor")
        floor = self.loader.loadModel("models/box")
        floor.setScale(size_x, size_y, 0.1)
        floor.setPos(0, 0, -0.05)
        floor.setColor(0.2, 0.2, 0.22, 1)
        floor.reparentTo(cm)

        # obstacles
        for b in self.mapdata.blocks:
            model = self.loader.loadModel("models/box")
            model.setScale(b.size[0], b.size[1], b.size[2])
            model.setPos(b.pos[0], b.pos[1], b.pos[2])
            model.setColor(0.6, 0.6, 0.6, 1)
            model.reparentTo(self.render)

        # bases / beacons
        self.red_beacon = self.loader.loadModel("models/box")
        self.red_beacon.setScale(2, 2, 6)
        self.red_beacon.setPos(self.mapdata.red_base[0], self.mapdata.red_base[1], 3)
        self.red_beacon.setColor(1.0, 0.5, 0.1, 1)
        self.red_beacon.reparentTo(self.render)

        self.blue_beacon = self.loader.loadModel("models/box")
        self.blue_beacon.setScale(2, 2, 6)
        self.blue_beacon.setPos(self.mapdata.blue_base[0], self.mapdata.blue_base[1], 3)
        self.blue_beacon.setColor(0.1, 0.5, 1.0, 1)
        self.blue_beacon.reparentTo(self.render)

        # beams root (cleared and rebuilt each frame)
        self.beam_group = self.render.attachNewNode("beams")

        # player representations
        self.player_nodes = {}  # pid -> NodePath
        self.grenade_nodes = {}  # gid -> NodePath
        self._grenade_hold_start = None
        self.explosion_nodes = []  # list of {"node": NodePath, "expire": float}

        # cosmetics
        self.local_team = None  # filled post-handshake
        self.cosmetics = CosmeticsManager(self)

        # camera init
        self.camera.setPos(0, -15, 3)
        self.mouse_locked = False
        self.center_mouse()

        # UI text (simple crosshair)
        from direct.gui.OnscreenText import OnscreenText
        self.crosshair = OnscreenText(text="+", pos=(0, 0), fg=(1, 1, 1, 1), scale=0.08, mayChange=True)
        # --- Killfeed HUD state ---
        hud_cfg = self.cfg.get("hud", {})
        self._kill_ttl   = float(hud_cfg.get("killfeed_ttl", 4.0))
        self._kill_max   = int(hud_cfg.get("killfeed_max", 6))
        self._kill_scale = float(hud_cfg.get("killfeed_font_scale", 0.045))

        self._kill_seen = set()            # (t_ms, attacker_pid, victim_pid)
        self._kill_nodes = []              # list[(OnscreenText, created_time)]

        # key state
        self.keys = set()
        for key in ["w", "a", "s", "d", "space", "control", "shift", "e", "tab", "m"]:
            self.accept(key, self.on_key, [key, True])
            self.accept(key + "-up", self.on_key, [key, False])
        self.accept("escape", sys.exit)

        # mouse update
        self.taskMgr.add(self.update_task, "update")

        # --- network start (connect first, then start recv loop) ---
        self.net_runner = AsyncRunner()

        def _after_connect(fut):
            try:
                fut.result()  # raise if connect() failed
            except Exception as e:
                print(f"[net] connect failed: {e}")
                return
            # record my team (used by CosmeticsManager to show teammate markers)
            self.local_team = self.client.team
            # start the receive loop once the reader/writer are ready
            self.net_runner.run_coro(self.client.recv_state_loop(self.on_state))

        connect_future = self.net_runner.run_coro(self.client.connect(host, port))
        connect_future.add_done_callback(_after_connect)

        # optional: a small heartbeat so you can see snapshots arriving
        self._last_state_log = 0.0

    def center_mouse(self):
        wp = WindowProperties()
        wp.setCursorHidden(True)
        self.win.requestProperties(wp)
        self.mouse_locked = True

    # --- Snapshot buffering & ingestion ---------------------------------

    def _insert_snapshot(self, snap: Dict[str, Any]):
        """
        Insert a snapshot into our time-sorted buffer.
        We keep ~2 seconds of history or at most ~120 snapshots.
        """
        t = float(snap.get("time", 0.0))
        idx = bisect(self.snap_times, t)
        self.snapshots.insert(idx, snap)
        self.snap_times.insert(idx, t)
        self.latest_server_time = max(self.latest_server_time, t)

        # Trim by max count
        MAX_SNAPS = 120
        if len(self.snapshots) > MAX_SNAPS:
            trim = len(self.snapshots) - MAX_SNAPS
            del self.snapshots[:trim]
            del self.snap_times[:trim]

        # Trim by age window (older than latest-2s)
        while len(self.snapshots) >= 2 and (self.snap_times[-1] - self.snap_times[0]) > 2.0:
            self.snapshots.pop(0)
            self.snap_times.pop(0)

    def on_state(self, state):
        # Thread-safe enough; render thread only reads.
        self._insert_snapshot(state)
        self.client.state = state  # still exposed if needed

        # Throttled log to confirm UI is being fed
        now = time.time()
        if now - getattr(self, "_last_state_log", 0) > 1.0:
            # print(f"[net] snapshot: t={state.get('time', 0):.2f} players={len(state.get('players', []))}")
            self._last_state_log = now

    # --- Input handling ---------------------------------------------------

    def on_key(self, key, down):
        if down:
            self.keys.add(key)
        else:
            self.keys.discard(key)

    def poll_mouse(self) -> Tuple[float, float]:
        if not self.mouseWatcherNode.hasMouse():
            return 0.0, 0.0
        m = self.mouseWatcherNode.getMouse()
        # convert to deltas by recentering each frame
        x = m.getX()
        y = m.getY()
        sens = self.cfg["controls"]["mouse_sensitivity"]
        dx, dy = x * sens * 100.0, y * sens * 100.0
        self.win.movePointer(0, int(self.win.getXSize() / 2), int(self.win.getYSize() / 2))
        return dx, dy

    # --- Interpolation helpers -------------------------------------------

    def _get_interp_pair(self) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """
        Choose two snapshots [s0, s1] bracketing our desired render time,
        and compute alpha in [0,1]. If we cannot bracket, we clamp.
        """
        if not self.snapshots:
            return None, None, 0.0
        if len(self.snapshots) == 1:
            # Only one snapshot: render as-is.
            s = self.snapshots[-1]
            return s, s, 0.0

        render_time = self.render_time

        # Find index i such that snap_times[i] <= render_time < snap_times[i+1]
        i = bisect_right(self.snap_times, render_time) - 1

        if i < 0:
            # Too early: clamp to first
            return self.snapshots[0], self.snapshots[0], 0.0
        if i >= len(self.snapshots) - 1:
            # Too late: clamp to last (avoid extrapolation)
            s = self.snapshots[-1]
            return s, s, 0.0

        t0 = self.snap_times[i]
        t1 = self.snap_times[i + 1]
        s0 = self.snapshots[i]
        s1 = self.snapshots[i + 1]
        if t1 <= t0 + 1e-6:
            return s1, s1, 0.0
        alpha = (render_time - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))
        return s0, s1, alpha

    # --- Render helpers ---------------------------------------------------

    def _render_beams(self, s0, s1, alpha):
        # Clear previous frame's beams
        for child in self.beam_group.getChildren():
            child.removeNode()
        if not s0 and not s1:
            return

        # Combine beams from the two snapshots we’re blending between
        beams0 = s0.get("beams", []) if s0 else []
        beams1 = s1.get("beams", []) if (s1 is not s0 and s1) else []
        beams = beams0 + beams1

        # Config
        vis = self.cfg.get("laser_visual", {})
        speed = float(vis.get("projectile_speed_mps", 180.0))
        streak_len = float(vis.get("streak_length_m", 2.0))
        thickness = float(vis.get("beam_thickness_px", 3.0))

        # Render clock (server-time) that our interpolation already uses
        render_time = self.render_time  # same clock as get_snapshots_for_render()

        col_red  = self.cfg["colors"]["team_red"]
        col_blue = self.cfg["colors"]["team_blue"]

        seen = set()
        for b in beams:
            key = (round(b["sx"],2), round(b["sy"],2), round(b["sz"],2),
                   round(b["ex"],2), round(b["ey"],2), round(b["ez"],2),
                   b.get("team", -1), int(b.get("t", 0.0) * 20))
            if key in seen:
                continue
            seen.add(key)

            # Time-of-flight progress
            t0 = float(b.get("t", 0.0))
            elapsed = render_time - t0
            if elapsed <= 0.0:
                continue

            sx, sy, sz = b["sx"], b["sy"], b["sz"]
            ex, ey, ez = b["ex"], b["ey"], b["ez"]
            dx, dy, dz = (ex - sx), (ey - sy), (ez - sz)
            L = float(b.get("len", math.sqrt(dx*dx + dy*dy + dz*dz)))
            if L <= 1e-6:
                continue

            head = min(L, elapsed * speed)
            tail = max(0.0, head - streak_len)

            if head <= 0.0:
                continue

            # Segment endpoints for the visible streak
            ux, uy, uz = (dx / L), (dy / L), (dz / L)
            x0 = sx + ux * tail
            y0 = sy + uy * tail
            z0 = sz + uz * tail
            x1 = sx + ux * head
            y1 = sy + uy * head
            z1 = sz + uz * head

            segs = LineSegs()
            segs.setThickness(thickness)
            if b.get("team") == TEAM_RED:
                segs.setColor(col_red[0], col_red[1], col_red[2], 1.0)
            else:
                segs.setColor(col_blue[0], col_blue[1], col_blue[2], 1.0)
            segs.moveTo(x0, y0, z0)
            segs.drawTo(x1, y1, z1)
            self.beam_group.attachNewNode(segs.create())

    def _clip_to_obstacles(self, x, y, z, radius=0.38, epsilon=0.02):
        """
        Keep (x,y) out of any block AABB that intersects height z by pushing it
        to the nearest boundary of the 'inflated' rectangle (inflated by radius).
        Iterates a couple passes to resolve corner overlaps.
        """
        changed = True
        iters = 0
        blocks = getattr(self.mapdata, "blocks", [])
        while changed and iters < 4:
            changed = False
            iters += 1
            for b in blocks:
                cx, cy, cz = b.pos
                sx, sy, sz = b.size
                hx = sx * 0.5 + radius
                hy = sy * 0.5 + radius
                hz = sz * 0.5
                # Only blocks that overlap the camera height matter
                if not (cz - hz <= z <= cz + hz):
                    continue
                dx = x - cx
                dy = y - cy
                if abs(dx) <= hx and abs(dy) <= hy:
                    # Inside → push out along least-penetration axis
                    push_x = (hx - abs(dx)) + epsilon
                    push_y = (hy - abs(dy)) + epsilon
                    if push_x < push_y:
                        x = cx + (hx if dx >= 0 else -hx)
                        x += (epsilon if dx >= 0 else -epsilon)
                    else:
                        y = cy + (hy if dy >= 0 else -hy)
                        y += (epsilon if dy >= 0 else -epsilon)
                    changed = True
        return x, y

    # --- Per-frame update -------------------------------------------------

    def _layout_killfeed(self):
        y = 0.92
        line = self._kill_scale * 1.6
        for node, _t in self._kill_nodes:
            node.setPos(0.95, y)  # top-right-ish; using right alignment below
            y -= line

    def _add_killfeed_item(self, text: str, color=(1,1,1,1)):
        node = OnscreenText(
            text=text, pos=(0.95, 0.92),  # will be re-laid
            fg=color, scale=self._kill_scale, align=TextNode.ARight, mayChange=True
        )
        # newest first
        self._kill_nodes.insert(0, (node, time.time()))
        # trim overflow
        while len(self._kill_nodes) > self._kill_max:
            old, _t = self._kill_nodes.pop()
            old.removeNode()
        self._layout_killfeed()

    def update_task(self, task):
        # advance smoothed render_time toward latest snapshot time
        target = self.latest_server_time - self.interp_delay + self.interp_predict
        dt = getattr(task, "dt", 0.0)
        if self.render_time == 0.0:
            self.render_time = target
        else:
            self.render_time = min(target, self.render_time + dt)
        # === Interpolate and render ===
        s0, s1, a = self._get_interp_pair()

        present = set()
        my_pid = self.client.pid

        def _players_by_pid(state):
            return {p["pid"]: p for p in state.get("players", [])} if state else {}

        d0 = _players_by_pid(s0)
        d1 = _players_by_pid(s1)

        union_pids = set(d0.keys()) | set(d1.keys())

        # Create/update nodes
        for pid in union_pids:
            p0 = d0.get(pid)
            p1 = d1.get(pid) if s1 is not s0 else d0.get(pid)

            if p0 is None and p1 is None:
                continue

            if p0 is None:
                # Only in later snapshot
                px, py, pz = p1["x"], p1["y"], p1["z"]
                roll = p1["roll"]
                pitch = p1["pitch"]
                yaw = p1["yaw"]
            elif p1 is None:
                # Only in earlier snapshot
                px, py, pz = p0["x"], p0["y"], p0["z"]
                roll = p0["roll"]
                pitch = p0["pitch"]
                yaw = p0["yaw"]
            else:
                # Interpolate positions & yaw
                px = (1 - a) * p0["x"] + a * p1["x"]
                py = (1 - a) * p0["y"] + a * p1["y"]
                pz = (1 - a) * p0["z"] + a * p1["z"]
                roll = _angle_lerp_deg(p0["roll"], p1["roll"], a)
                pitch = _angle_lerp_deg(p0["pitch"], p1["pitch"], a)
                yaw = _angle_lerp_deg(p0["yaw"], p1["yaw"], a)

            # Ensure a node exists
            node = self.player_nodes.get(pid)
            if node is None:
                node = self.loader.loadModel("models/box.egg")
                pr = float(self.cfg.get("gameplay", {}).get("player_radius", 0.5))
                ph = float(self.cfg.get("gameplay", {}).get("player_height", 2.0))
                node.setScale(2.0 * pr, 2.0 * pr, ph)
                team = (p1 or p0)["team"]
                col = (1.0, 0.5, 0.1, 1) if team == TEAM_RED else (0.1, 0.5, 1.0, 1)
                node.setColor(*col)
                node.reparentTo(self.render)
                self.player_nodes[pid] = node

                # attach cosmetics (headgear, nameplate, teammate caret)
                name = (p1 or p0).get("name", f"PID{pid}")
                self.cosmetics.attach(pid, node, name, team)

            node.setPos(px, py, pz)
            node.setHpr(yaw, pitch, roll)

            alive_state = bool(((p1 or p0) or {}).get("alive", True))
            if not alive_state:
                node.setRenderModeWireframe()
            else:
                node.setRenderModeFilled()

            present.add(pid)

        # Remove nodes for players no longer present
        for pid in list(self.player_nodes.keys()):
            if pid not in present:
                # remove cosmetics first to avoid orphaned overlays
                if hasattr(self, "cosmetics"):
                    self.cosmetics.detach(pid)
                self.player_nodes[pid].removeNode()
                del self.player_nodes[pid]

        # --- Grenades ---
        g_present = set()
        latest = s1 or s0
        if latest:
            for g in latest.get("grenades", []):
                gid = g.get("id")
                node = self.grenade_nodes.get(gid)
                if node is None:
                    node = self.loader.loadModel("models/box")
                    node.setScale(0.4)
                    col = (1, 0, 0, 1) if g.get("team") == TEAM_RED else (0, 0, 1, 1)
                    node.setColor(*col)
                    node.reparentTo(self.render)
                    self.grenade_nodes[gid] = node
                node.setPos(g.get("x",0.0), g.get("y",0.0), g.get("z",0.0))
                g_present.add(gid)
        for gid in list(self.grenade_nodes.keys()):
            if gid not in g_present:
                self.grenade_nodes[gid].removeNode()
                del self.grenade_nodes[gid]


        if latest:
            for e in latest.get("explosions", []):
                node = self.loader.loadModel("models/box")
                radius = float(self.cfg.get("gameplay", {}).get("grenade_radius", 5.0))
                node.setScale(radius)
                node.setColor(1, 0.5, 0, 0.7)
                node.setTransparency(TransparencyAttrib.M_alpha)
                node.setPos(e.get("x",0.0), e.get("y",0.0), e.get("z",0.0))
                node.reparentTo(self.render)
                self.explosion_nodes.append({"node": node, "expire": self.render_time + 0.3})
        for ent in list(self.explosion_nodes):
            if self.render_time >= ent["expire"]:
                ent["node"].removeNode()
                self.explosion_nodes.remove(ent)

        # Smooth camera **position** using our interpolated "me" (orientation from live mouse)
        if my_pid is not None:
            me0 = d0.get(my_pid)
            me1 = d1.get(my_pid) if s1 is not s0 else d0.get(my_pid)
            if me0 or me1:
                if me0 is None:
                    cx, cy, cz = me1["x"], me1["y"], me1["z"]
                elif me1 is None:
                    cx, cy, cz = me0["x"], me0["y"], me0["z"]
                else:
                    cx = (1 - a) * me0["x"] + a * me1["x"]
                    cy = (1 - a) * me0["y"] + a * me1["y"]
                    cz = (1 - a) * me0["z"] + a * me1["z"]

                # Place camera near head height
                ph = float(self.cfg.get("gameplay", {}).get("player_height", 2.0))
                cz = cz + 0.30 * ph  # ~eye height

                # === NEW: obstacle-aware camera clipping so we can't see inside blocks ===
                # Prefer configured radius if present; fallback to ~player capsule radius.
                try:
                    radius = float(self.cfg.get("gameplay", {}).get("player_radius", 0.5))
                except Exception:
                    radius = 0.5
                cx, cy = self._clip_to_obstacles(cx, cy, cz, radius=radius, epsilon=0.02)

                self.camera.setPos(cx, cy, cz)

        # draw recent beams from the current snapshots
        self._render_beams(s0, s1, a)

        # update cosmetics (distance fade + billboards)
        if hasattr(self, "cosmetics"):
            self.cosmetics.update()

        # --- Killfeed ingest & prune ---
        latest = s1 or s0
        if latest:
            feed = latest.get("killfeed", [])
            for ev in feed:
                key = (int(float(ev.get("t", 0.0)) * 1000), ev.get("attacker"), ev.get("victim"))
                if key in self._kill_seen:
                    continue
                self._kill_seen.add(key)
                msg = f"{ev.get('attacker_name','?')} eliminated {ev.get('victim_name','?')}"
                self._add_killfeed_item(msg)

        # expire old killfeed entries (no fade, simple TTL)
        nowt = time.time()
        for i in range(len(self._kill_nodes) - 1, -1, -1):
            node, t0 = self._kill_nodes[i]
            if (nowt - t0) >= self._kill_ttl:
                node.removeNode()
                self._kill_nodes.pop(i)
        if self._kill_nodes:
            self._layout_killfeed()

        # === Build & send inputs ===
        mx = 0.0
        mz = 0.0
        if "w" in self.keys:
            mz += 1
        if "s" in self.keys:
            mz -= 1
        if "a" in self.keys:
            mx -= 1
        if "d" in self.keys:
            mx += 1

        # mouse deltas → local yaw/pitch (degrees)
        dx, dy = self.poll_mouse()
        invert = bool(self.cfg["controls"].get("invert_y", False))
        self.yaw += -dx
        self.pitch += (-dy if invert else dy)
        self.pitch = max(-90.0, min(90.0, self.pitch))

        # apply camera orientation (HPR). Position is handled above via interpolation.
        self.camera.setHpr(self.yaw, self.pitch, 0)

        data = {
            "mx": mx,
            "mz": mz,
            "jump": "space" in self.keys,
            "crouch": "control" in self.keys,
            "walk": "shift" in self.keys,
            "fire": False,
            "interact": "e" in self.keys,
            "yaw": self.yaw,
            "pitch": self.pitch,
        }

        if self.mouseWatcherNode.is_button_down(MouseButton.one()):
            data["fire"] = True
            data["fire_t"] = self.render_time

        # Middle mouse for grenade throw
        if self.mouseWatcherNode.is_button_down(MouseButton.two()):
            if self._grenade_hold_start is None:
                self._grenade_hold_start = self.render_time
        elif self._grenade_hold_start is not None:
            hold = max(0.0, self.render_time - self._grenade_hold_start)
            data["grenade"] = hold
            self._grenade_hold_start = None

        if self.client.writer:
            self.net_runner.run_coro(self.client.send_input(data))

        return task.cont

def pick_server_via_discovery(lan_port: int):
    # blocking helper to run discovery once using asyncio
    loop = asyncio.get_event_loop()
    servers = loop.run_until_complete(lan_discovery_broadcast(lan_port, timeout=1.0))
    if not servers:
        return None, None
    s = servers[0]
    return s["addr"], s["tcp_port"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--name", default="Player")
    ap.add_argument("--host", default=None, help="Server host; omit to auto-discover on LAN")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--interp_delay",
        type=float,
        default=0.10,
        help="Seconds of render-time interpolation delay buffer (e.g., 0.10 = 100 ms)",
    )
    ap.add_argument(
        "--interp_predict",
        type=float,
        default=0.0,
        help="Seconds of render-time prediction/extrapolation",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    host, port = args.host, args.port
    if not host:
        host, port = pick_server_via_discovery(cfg["server"]["lan_discovery_port"])
        if not host:
            print("No LAN servers discovered. Provide --host and --port.")
            sys.exit(1)
    if not port:
        port = cfg["server"]["port"]

    app = GameApp(cfg, host=host, port=port, name=args.name, interp_delay=args.interp_delay, interp_predict=args.interp_predict)
    app.run()


if __name__ == "__main__":
    main()
