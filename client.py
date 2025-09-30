# client.py
import sys, asyncio, json, time, math, argparse, threading, random, os
from typing import Dict, Any, List, Tuple
from bisect import bisect, bisect_right

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import Vec3, Point3, DirectionalLight, AmbientLight, LVector3f, KeyboardButton, WindowProperties, ClockObject
from panda3d.core import LColor, MouseButton, LineSegs, TextNode
from panda3d.core import TextNode, TransparencyAttrib, NodePath, LVecBase4f
from panda3d.core import TextProperties, TextPropertiesManager


class HudFeed:
    def __init__(self, x: float, y_start: float, align, scale: float, max_items: int, ttl: float):
        self.x = x
        self.y_start = y_start
        self.align = align
        self.scale = float(scale)
        self.max_items = int(max_items)
        self.ttl = float(ttl)
        self._nodes = []  # list[(OnscreenText, created_time)]

    def _compose_markup(self, parts):
        # parts: list of (text, team_or_None)
        out = []
        for txt, team in parts:
            if team == TEAM_RED:
                out.append("\x01tr\x01" + str(txt) + "\x02")
            elif team == TEAM_BLUE:
                out.append("\x01tb\x01" + str(txt) + "\x02")
            else:
                out.append(str(txt))
        return "".join(out)

    def add_parts(self, parts):
        node = OnscreenText(
            text=self._compose_markup(parts), pos=(self.x, self.y_start),
            fg=(1,1,1,1), scale=self.scale, align=self.align, mayChange=True
        )
        self._nodes.insert(0, (node, time.time()))
        while len(self._nodes) > self.max_items:
            old, _t = self._nodes.pop()
            old.removeNode()
        self._layout()

    def _layout(self):
        y = self.y_start
        line = self.scale * 1.6
        for node, _t in self._nodes:
            node.setPos(self.x, y)
            y -= line

    def prune(self, now_t: float):
        changed = False
        for i in range(len(self._nodes) - 1, -1, -1):
            node, t0 = self._nodes[i]
            if (now_t - t0) >= self.ttl:
                node.removeNode()
                self._nodes.pop(i)
                changed = True
        if changed and self._nodes:
            self._layout()
from panda3d.core import MaterialAttrib, ColorScaleAttrib, Material
from panda3d.core import CompassEffect, BillboardEffect, LColor
from panda3d.core import GeomNode, GeomVertexReader, GeomVertexWriter, GeomTriangles, GeomVertexFormat
from panda3d.core import loadPrcFileData

from common.net import send_json, read_json, lan_discovery_broadcast
from game.constants import TEAM_RED, TEAM_BLUE, TEAM_NEUTRAL
from scoreboard import Scoreboard
from game.map_gen import load_from_file as load_map_from_file

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
        # Prefer client-side overrides if provided
        try:
            self.cfg = dict(client.cfg.get("cosmetics", {}))
            settings_cos = (client.settings or {}).get("cosmetics", {}) if hasattr(client, "settings") else {}
            if settings_cos:
                # shallow merge; nested blocks handled as whole
                for k, v in settings_cos.items():
                    self.cfg[k] = v
            # If profile headgear preference exists, set as default for this client name
            prof = (client.settings or {}).get("profile", {}) if hasattr(client, "settings") else {}
            hg_pref = prof.get("headgear") if isinstance(prof, dict) else None
            if isinstance(hg_pref, dict):
                self.cfg.setdefault("headgear", {}).setdefault("default", {})
                self.cfg["headgear"]["default"].update(hg_pref)
        except Exception:
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

        # Try to attach to a model-provided anchor named "Hat" if present
        hat_anchor = None
        try:
            hat_anchor = player_np.find("**/Hat")
            if hat_anchor.isEmpty():
                hat_anchor = None
        except Exception:
            hat_anchor = None

        if hat_anchor is not None:
            # Parent cosmetics under the model's Hat node so they inherit its transform
            root = hat_anchor.attachNewNode(f"cosmetics-{pid}")
            head_anchor = root  # use root as the anchor in this case
        else:
            # Fallback: create a simple anchor above the player's origin
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


def load_settings(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[settings] failed to load {path}: {e}")
        return {}


def apply_prc_from_settings(settings: Dict[str, Any]):
    """Apply Panda3D PRC (engine) settings before ShowBase init."""
    vid = settings.get("video", {}) if settings else {}
    aud = settings.get("audio", {}) if settings else {}
    # Window size and fullscreen
    try:
        res = vid.get("resolution")
        if isinstance(res, list) and len(res) == 2:
            w, h = int(res[0]), int(res[1])
            loadPrcFileData("client-video", f"win-size {w} {h}")
    except Exception:
        pass
    try:
        if "fullscreen" in vid:
            fs = bool(vid.get("fullscreen", False))
            loadPrcFileData("client-video", f"fullscreen {'#t' if fs else '#f'}")
    except Exception:
        pass
    try:
        if "vsync" in vid:
            vs = bool(vid.get("vsync", True))
            loadPrcFileData("client-video", f"sync-video {1 if vs else 0}")
    except Exception:
        pass
    # Audio device hint (must be set pre-initialization)
    try:
        dev = aud.get("device")
        if isinstance(dev, str) and dev:
            loadPrcFileData("client-audio", f"audio-device {dev}")
    except Exception:
        pass


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
    def __init__(self, cfg, name="Player", team_pref: str = "auto"):
        self.cfg = cfg
        self.name = name
        self.team_pref = team_pref
        self.reader = None
        self.writer = None
        self.pid = None
        self.team = None
        self.state = None
        self.last_input = {}

    async def connect(self, host: str, port: int):
        self.reader, self.writer = await asyncio.open_connection(host, port)
        await send_json(self.writer, {"type": "hello", "name": self.name, "team_preference": self.team_pref})
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
        await send_json(self.writer, {"type": "input", "time": time.time(), "data": data})


def _angle_lerp_deg(a: float, b: float, t: float) -> float:
    """
    Shortest-path spherical (mod 360) interpolation for degrees.
    """
    # Map delta into [-180, 180]
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + d * t


class GameApp(ShowBase):
    def __init__(self, cfg, host: str, port: int, name: str, interp_delay: float, interp_predict: float = 0.0, settings: Dict[str, Any] = None, team_pref: str = "auto"):
        ShowBase.__init__(self)
        self.set_background_color(0.05, 0.05, 0.07, 1)
        self.disableMouse()

        self.settings = settings or {}

        cam_cfg = cfg.get("camera", {})
        near = float(cam_cfg.get("near", 0.03))
        far  = float(self.settings.get("video", {}).get("render_distance", cam_cfg.get("far", 300.0)))
        base.camLens.setNear(near)
        base.camLens.setFar(far)
        # Optional FOV override
        try:
            fov = self.settings.get("controls", {}).get("fov")
            if fov is not None:
                base.camLens.setFov(float(fov))
        except Exception:
            pass

        self.cfg = cfg
        self.client = NetworkClient(cfg, name=name, team_pref=team_pref)
        self.host, self.port = host, port
        self.yaw, self.pitch = 0.0, 0.0
        # Controls overrides (mouse_sensitivity, invert_y, toggle_crouch)
        self._controls = dict(cfg.get("controls", {}))
        try:
            self._controls.update(self.settings.get("controls", {}))
        except Exception:
            pass
        self._invert_y = bool(self._controls.get("invert_y", False))
        self._toggle_crouch = bool(self._controls.get("toggle_crouch", False))

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

        # arena must come from a map file
        map_cfg = dict(cfg.get("map", {}))
        map_file = map_cfg.get("file")
        if not isinstance(map_file, str) or not map_file:
            raise ValueError("Configuration must provide 'map.file' with a valid JSON map path")

        try:
            self.mapdata = load_map_from_file(map_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load map file '{map_file}': {e}") from e

        size_x, size_y = self.mapdata.bounds
        cfg["gameplay"]["arena_size_m"] = [size_x, size_y]
        print(f"[map] Loaded '{map_file}' ({size_x:.1f}×{size_y:.1f} m)")

        # floor plane (visual only)
        cm = self.render.attachNewNode("floor")
        floor = self.loader.loadModel("models/box")
        floor.setScale(size_x, size_y, 0.1)
        floor.setPos(0, 0, -0.05)
        floor.setColor(0.2, 0.2, 0.22, 1)
        floor.reparentTo(cm)

        # --- Obstacles: glTF cube + simplepbr with per-instance UV remap ---
        # Enable simplepbr for proper glTF PBR if available
        self._pbr = None
        try:
            import simplepbr
            self._pbr = simplepbr.init(use_normal_maps=True, max_lights=8)
            try:
                self._pbr.exposure = float((self.settings or {}).get("video", {}).get("pbr_exposure", -0.5))
            except Exception:
                pass
            print("[obstacles] simplepbr pipeline active")
        except Exception as e:
            print(f"[obstacles] simplepbr not active: {e}")

        # Load cube.glb via python gltf loader
        self.cube_template = None
        try:
            import gltf
            self.cube_template = gltf.load_model("models/cube.glb")
            if not hasattr(self.cube_template, 'findAllMatches'):
                self.cube_template = NodePath(self.cube_template)
        except Exception as e:
            print(f"[obstacles] Failed to load models/cube.glb: {e}")
            self.cube_template = None

        def _tile_pair_from(val, grid):
            # Accept either a single index or a [tx,ty] pair
            try:
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    tx = int(val[0]); ty = int(val[1])
                    return (max(0, min(grid[0]-1, tx)), max(0, min(grid[1]-1, ty)))
            except Exception:
                pass
            try:
                idx = int(val)
            except Exception:
                idx = 0
            # Default 2x2 mapping
            return (idx % grid[0], idx // grid[0])

        def _atlas_cfg():
            # Pull mapping from config if present
            try:
                return dict(self.cfg.get('cube_atlas', {}))
            except Exception:
                return {}

        def _face_tiles_for_type(t: int):
            atlas = _atlas_cfg()
            grid = atlas.get('grid', [2,2])
            types = atlas.get('types', {})
            spec = types.get(str(int(t))) or types.get(int(t))
            if spec is None:
                # Fallback defaults
                defaults = {
                    '0': { 'all': 0, 'top': 2 },
                    '1': { 'all': 1, 'top': 3 },
                    '2': { 'all': 2 },
                    '3': { 'all': 3 },
                }
                spec = defaults.get(str(int(t)), defaults['0'])
            faces = {}
            if 'all' in spec:
                pair = _tile_pair_from(spec['all'], grid)
                for f in ('+x','-x','+y','-y','+z','-z'):
                    faces[f] = pair
            for k,v in spec.items():
                if k == 'all':
                    continue
                key = k if str(k).startswith(('+','-')) else str(k)
                faces[key] = _tile_pair_from(v, grid)
            return faces

        def _uv_remap_to_tiles(root: NodePath, face_tiles, grid=(2,2)):
            # duplicate geoms so this instance is unique
            col = root.findAllMatches('**/+GeomNode')
            for i in range(col.getNumPaths()):
                gnp = col.getPath(i)
                gnode = gnp.node()
                for gi in range(gnode.getNumGeoms()):
                    gnode.setGeom(gi, gnode.getGeom(gi).makeCopy())

            aliases = {'right': '+x', 'left': '-x', 'front': '+y', 'back': '-y', 'top': '+z', 'bottom': '-z'}
            wanted = {}
            for k, v in face_tiles.items():
                kk = aliases.get(str(k).lower(), str(k).lower())
                wanted[kk] = v

            tile_w, tile_h = 1.0/grid[0], 1.0/grid[1]

            def classify(p0, p1, p2):
                n = (p1 - p0).cross(p2 - p0)
                if n.lengthSquared() == 0: return 'unknown'
                n.normalize()
                ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
                if ax >= ay and ax >= az: return '+x' if n.x>0 else '-x'
                if ay >= ax and ay >= az: return '+y' if n.y>0 else '-y'
                return '+z' if n.z>0 else '-z'

            for i in range(col.getNumPaths()):
                gnp = col.getPath(i)
                gnode = gnp.node()
                for gi in range(gnode.getNumGeoms()):
                    geom = gnode.modifyGeom(gi)
                    vdata = geom.modifyVertexData()
                    # discover UV column
                    uv_name = None
                    fmt = vdata.getFormat()
                    try:
                        for ai in range(fmt.getNumArrays()):
                            arr = fmt.getArray(ai)
                            for ci in range(arr.getNumColumns()):
                                nm = arr.getColumn(ci).getName().getName()
                                if 'texcoord' in nm.lower():
                                    uv_name = nm; raise StopIteration
                    except StopIteration:
                        pass
                    if uv_name is None:
                        continue
                    rdr_v = GeomVertexReader(vdata, 'vertex')
                    rdr_uv = GeomVertexReader(vdata, uv_name)
                    wtr_uv = GeomVertexWriter(vdata, uv_name)
                    for p in range(geom.getNumPrimitives()):
                        prim = geom.modifyPrimitive(p).decompose()
                        geom.setPrimitive(p, prim)
                        nverts = prim.getNumVertices()
                        for vi in range(0, nverts, 3):
                            i0 = prim.getVertex(vi)
                            i1 = prim.getVertex(vi+1)
                            i2 = prim.getVertex(vi+2)
                            rdr_v.setRow(i0); p0 = rdr_v.getData3f()
                            rdr_v.setRow(i1); p1 = rdr_v.getData3f()
                            rdr_v.setRow(i2); p2 = rdr_v.getData3f()
                            face = classify(p0,p1,p2)
                            if face not in wanted:
                                continue
                            tx, ty = wanted[face]
                            off_u, off_v = tx*tile_w, ty*tile_h
                            for idx in (i0,i1,i2):
                                rdr_uv.setRow(idx); u,v = rdr_uv.getData2f()
                                wtr_uv.setRow(idx); wtr_uv.setData2f(off_u + u*tile_w, off_v + v*tile_h)

        if self.cube_template is not None:
            # Pre-bake a small set of variants (0..3) by remapping UVs once per type,
            # then instance those variants for each block. This avoids rewriting
            # vertex data for every single block.
            self.cube_variants = {}
            for t in range(4):
                base_np = self.cube_template.copyTo(NodePath('cube_variants'))
                try:
                    _uv_remap_to_tiles(base_np, _face_tiles_for_type(t))
                except Exception as e:
                    print(f"[obstacles] variant {t} UV remap failed: {e}")
                self.cube_variants[t] = base_np
            print('[obstacles] prepared cube variants for types 0..3')

            # Precompute team tint config and colors
            atl_cfg = self.cfg.get('cube_atlas', {}) if isinstance(self.cfg, dict) else {}
            tt_cfg = atl_cfg.get('team_tint', {}) if isinstance(atl_cfg, dict) else {}
            tt_enabled = bool(tt_cfg.get('enabled', True))
            tt_intensity = float(tt_cfg.get('intensity', 0.08))
            tt_blend = float(tt_cfg.get('blend_width_m', 8.0))
            try:
                col_red = self.cfg.get('colors', {}).get('team_red', [0.85,0.2,0.1,1.0])
                col_blue = self.cfg.get('colors', {}).get('team_blue', [0.15,0.45,0.95,1.0])
            except Exception:
                col_red = [0.85,0.2,0.1,1.0]
                col_blue = [0.15,0.45,0.95,1.0]

            first = True
            for b in self.mapdata.blocks:
                t = int(getattr(b, 'box_type', 0)) % 4
                src = self.cube_variants.get(t, self.cube_template)
                inst = src.copyTo(self.render)
                inst.setPos(b.pos[0], b.pos[1], b.pos[2])
                sx, sy, sz = b.size
                if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6) or (abs(sz - 1.0) > 1e-6):
                    inst.setScale(sx, sy, sz)
                # Subtle team tint based on X half, with blend across the center line
                try:
                    if tt_enabled and tt_intensity > 0.0:
                        x = b.pos[0]
                        # Weight ramps from 0 near center to 1 beyond blend half-width
                        w = 1.0
                        if tt_blend > 0.0:
                            half = 0.5 * tt_blend
                            w = min(1.0, max(0.0, (abs(x) - 0.0) / max(half, 1e-6)))
                        team = col_blue if x >= 0 else col_red
                        sr = (1.0 - tt_intensity * w) + tt_intensity * w * float(team[0])
                        sg = (1.0 - tt_intensity * w) + tt_intensity * w * float(team[1])
                        sb = (1.0 - tt_intensity * w) + tt_intensity * w * float(team[2])
                        inst.setColorScale(sr, sg, sb, 1.0)
                except Exception:
                    pass
                if first:
                    try:
                        print('[obstacles] first instance textures:', [t.getName() for t in inst.findAllTextures()], 'type:', getattr(b,'box_type',0))
                    except Exception:
                        pass
                    first = False

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

        # --- Player/Bot visuals ---
        # Try to load the robot GLB once (requires panda3d-gltf)
        self.robot_template = None
        self._robot_bounds = None  # (min_v, max_v) in template local space
        try:
            self.robot_template = self.loader.loadModel("models/robot.glb")
            b = self.robot_template.getTightBounds()
            if b is not None:
                self._robot_bounds = (b[0], b[1])
        except Exception as e:
            print(f"[init] Could not load models/robot.glb: {e}. Falling back to box.")
            self.robot_template = None

        # player representations
        self.player_nodes = {}  # pid -> NodePath
        self.robot_parts = {}   # pid -> dict(root, model, base, body, head, shoulder_r, blaster)
        self._local_robot_root = None  # hidden local player model root for anchors
        self._prev_pos = {}     # pid -> (x, y)
        self._last_base_h = {}  # pid -> float last computed base heading
        self.grenade_nodes = {}  # gid -> NodePath
        self._grenade_hold_start = None
        self.explosion_nodes = []  # list of {"node": NodePath, "expire": float}

        # flag visuals
        self.flag_template = None
        self.flag_nodes: Dict[int, NodePath] = {}  # key: team id (TEAM_NEUTRAL or team flag id) -> NodePath
        self._flag_bounds = None  # (min_v, max_v) if loaded
        try:
            self.flag_template = self.loader.loadModel("models/flag.glb")
            b = self.flag_template.getTightBounds()
            if b is not None:
                self._flag_bounds = (b[0], b[1])
        except Exception as e:
            print(f"[init] Could not load models/flag.glb: {e}. Using box fallback.")
            self.flag_template = self.loader.loadModel("models/box")
        # determine neutral color (distinct from teams)
        try:
            col = self.cfg.get("colors", {}).get("neutral_flag")
            if col is None:
                # bright gold/yellow default (distinct from red/blue)
                col = [1.0, 0.92, 0.25, 1.0]
            self._neutral_flag_color = LColor(*col)
        except Exception:
            self._neutral_flag_color = LColor(1.0, 0.92, 0.25, 1.0)

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
        # Optional debug netgraph
        self._netgraph_enabled = bool((self.settings or {}).get("debug", {}).get("netgraph", False))
        self._netgraph = None
        if self._netgraph_enabled:
            self._netgraph = OnscreenText(text="", pos=(-1.3, 0.95), fg=(0.8, 1.0, 0.8, 1), align=TextNode.ALeft, scale=0.04, mayChange=True)
        # --- Killfeed/Message HUD state ---
        hud_cfg = self.cfg.get("hud", {})
        self._kill_ttl   = float(hud_cfg.get("killfeed_ttl", 4.0))
        self._kill_max   = int(hud_cfg.get("killfeed_max", 6))
        self._kill_scale = float(hud_cfg.get("killfeed_font_scale", 0.045))

        self._kill_seen = set()            # (t_ms, attacker_pid, victim_pid)
        # Message feed (top-left) — similar policy to killfeed
        self._msg_ttl   = float(hud_cfg.get("messagefeed_ttl", self._kill_ttl))
        self._msg_max   = int(hud_cfg.get("messagefeed_max", self._kill_max))
        self._msg_scale = float(hud_cfg.get("messagefeed_font_scale", self._kill_scale))
        self._msg_seen = set()             # (t_ms, actor_pid, event)

        # Text styling: define properties for team-colored inline segments
        try:
            tpm = TextPropertiesManager.getGlobalPtr()
            colors = self.cfg.get("colors", {})
            col_r = colors.get("team_red", (1.0, 0.25, 0.25, 1.0))
            col_b = colors.get("team_blue", (0.25, 0.5, 1.0, 1.0))
            tp_r = TextProperties(); tp_r.setTextColor(*col_r)
            tp_b = TextProperties(); tp_b.setTextColor(*col_b)
            tpm.setProperties("tr", tp_r)
            tpm.setProperties("tb", tp_b)
        except Exception:
            pass

        # Initialize feeds using a shared component
        self.kill_feed = HudFeed(x=0.95, y_start=0.92, align=TextNode.ARight,
                                 scale=self._kill_scale, max_items=self._kill_max, ttl=self._kill_ttl)
        self.msg_feed  = HudFeed(x=-1.25, y_start=0.92, align=TextNode.ALeft,
                                 scale=self._msg_scale, max_items=self._msg_max, ttl=self._msg_ttl)
        # Message feed (top-left) — similar policy to killfeed
        self._msg_ttl   = float(hud_cfg.get("messagefeed_ttl", self._kill_ttl))
        self._msg_max   = int(hud_cfg.get("messagefeed_max", self._kill_max))
        self._msg_scale = float(hud_cfg.get("messagefeed_font_scale", self._kill_scale))
        self._msg_seen = set()             # (t_ms, actor_pid, event)

        # Text styling: define properties for team-colored inline segments
        try:
            tpm = TextPropertiesManager.getGlobalPtr()
            colors = self.cfg.get("colors", {})
            col_r = colors.get("team_red", (1.0, 0.25, 0.25, 1.0))
            col_b = colors.get("team_blue", (0.25, 0.5, 1.0, 1.0))
            tp_r = TextProperties(); tp_r.setTextColor(*col_r)
            tp_b = TextProperties(); tp_b.setTextColor(*col_b)
            tpm.setProperties("tr", tp_r)
            tpm.setProperties("tb", tp_b)
        except Exception:
            pass
        gp_cfg = self.cfg.get("gameplay", {})
        self.shots_per_mag = int(gp_cfg.get("shots_per_mag", 20))
        self.reload_seconds = float(gp_cfg.get("reload_seconds", 1.5))
        self.rapid_fire_rate = float(gp_cfg.get("rapid_fire_rate_hz", 10.0))
        # Recoil model (client-side aim offsets only)
        # Use simplified gameplay keys; ignore legacy camera_recoil block.
        self.recoil_per_shot_deg = float(gp_cfg.get("recoil_per_shot_deg", 0.1))
        # Exponential decay rate in Hz (larger = faster return). Fallback preserves behavior if key absent.
        self.recoil_decay_hz = float(gp_cfg.get("recoil_decay_hz", 7.5))
        # Recoil offsets (degrees) applied to aim only, not camera/view
        self.aim_yaw_offset = 0.0
        self.aim_pitch_offset = 0.0
        # Track firing/reload state to control when offsets reset
        self.prev_fire_pressed = False
        self.was_reloading = False
        self.shots_left = self.shots_per_mag
        self.reload_end = 0.0
        self.last_local_fire = 0.0
        self.ammo_text = OnscreenText(text=str(self.shots_left), pos=(1.25, -0.95), fg=(1,1,1,1), align=TextNode.ARight, scale=0.07, mayChange=True)
        self.scoreboard = Scoreboard(self)

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
        # Support toggle crouch via settings
        if key == "control" and self._toggle_crouch:
            if down:
                self._crouch_toggle_state = not bool(getattr(self, "_crouch_toggle_state", False))
            return
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
        sens = float(self._controls.get("mouse_sensitivity", self.cfg.get("controls", {}).get("mouse_sensitivity", 0.12)))
        dx, dy = x * sens * 100.0, y * sens * 100.0
        self.win.movePointer(0, int(self.win.getXSize() / 2), int(self.win.getYSize() / 2))
        return dx, dy


    def _apply_recoil(self):
        # Accumulate recoil into aim offsets (do not move the camera)
        # Apply a fixed vertical kick per shot. No yaw jitter by default.
        self.aim_pitch_offset -= float(self.recoil_per_shot_deg)

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

            ex, ey, ez = b["ex"], b["ey"], b["ez"]
            sx_orig, sy_orig, sz_orig = b["sx"], b["sy"], b["sz"]
            L_orig = float(b.get("len", math.sqrt((ex - sx_orig)**2 + (ey - sy_orig)**2 + (ez - sz_orig)**2)))
            # Try to replace start with the shooter's Blaster anchor if we can
            sx, sy, sz = sx_orig, sy_orig, sz_orig
            owner = b.get("owner")
            if owner is not None:
                parts = self.robot_parts.get(owner)
                if parts is not None:
                    blaster = parts.get("blaster")
                    try:
                        if blaster is not None and not blaster.isEmpty():
                            pos_w = blaster.getPos(self.render)
                            sx, sy, sz = float(pos_w.x), float(pos_w.y), float(pos_w.z)
                    except Exception:
                        pass

            dx, dy, dz = (ex - sx), (ey - sy), (ez - sz)
            L_new = math.sqrt(dx*dx + dy*dy + dz*dz)
            if L_new <= 1e-6:
                continue

            head = min(L_orig, elapsed * speed)
            tail = max(0.0, head - streak_len)

            if head <= 0.0:
                continue

            # Segment endpoints for the visible streak
            ux, uy, uz = (dx / L_new), (dy / L_new), (dz / L_new)
            head_c = min(head, L_new)
            x0 = sx + ux * tail
            y0 = sy + uy * tail
            z0 = sz + uz * tail
            x1 = sx + ux * head_c
            y1 = sy + uy * head_c
            z1 = sz + uz * head_c

            segs = LineSegs()
            segs.setThickness(thickness)
            if b.get("team") == TEAM_RED:
                segs.setColor(col_red[0], col_red[1], col_red[2], 1.0)
            else:
                segs.setColor(col_blue[0], col_blue[1], col_blue[2], 1.0)
            segs.moveTo(x0, y0, z0)
            segs.drawTo(x1, y1, z1)
            self.beam_group.attachNewNode(segs.create())

    def _get_flag_color(self, team: int) -> LColor:
        colors = self.cfg.get("colors", {})
        if team == TEAM_RED:
            r, g, b, a = colors.get("team_red", (1.0, 0.25, 0.25, 1.0))
            return LColor(r, g, b, a)
        if team == TEAM_BLUE:
            r, g, b, a = colors.get("team_blue", (0.25, 0.5, 1.0, 1.0))
            return LColor(r, g, b, a)
        return self._neutral_flag_color

    def _animate_robot_parts(self, pid: int, parts: Dict[str, NodePath], yaw: float, pitch: float, roll: float,
                              p0: Dict[str, Any], p1: Dict[str, Any], s0: Dict[str, Any], s1: Dict[str, Any], a: float,
                              px: float, py: float):
        """Animate model subparts by name according to movement and view.

        Root (player node) already has yaw applied.
        - Base: local heading (H) aligns with velocity direction in world, i.e. H = base_heading_world - yaw.
        - Head: local pitch (P) follows view pitch; local heading stays 0 (inherits root yaw).
        - Shoulder.R: local pitch (P) follows view pitch.
        """
        # Compute velocity direction from snapshots if possible, else from last pos
        vx = vy = 0.0
        if p0 is not None and p1 is not None and (p1 is not p0):
            try:
                vx = float(p1["x"]) - float(p0["x"]) ; vy = float(p1["y"]) - float(p0["y"])
            except Exception:
                vx = vy = 0.0
        if abs(vx) < 1e-5 and abs(vy) < 1e-5:
            prev = self._prev_pos.get(pid)
            if prev is not None:
                vx = px - prev[0]; vy = py - prev[1]
        # Update last pos cache
        self._prev_pos[pid] = (px, py)

        # Determine base heading in world (degrees). If stationary, reuse last or fall back to yaw.
        moving = (vx*vx + vy*vy) > 1e-6
        if moving:
            base_h = math.degrees(math.atan2(vy, vx))  # Panda3D: H=atan2(x, y)
            self._last_base_h[pid] = base_h
        else:
            base_h = self._last_base_h.get(pid, yaw)

        base_np = parts.get("base")
        if base_np is not None:
            try:
                # Convert world heading to local by subtracting root yaw
                local_h = base_h - yaw
                base_np.setHpr(local_h, 0.0, 0.0)
            except Exception:
                pass

        # Head follows yaw (inherited via Body) and pitch directly
        head_np = parts.get("head")
        if head_np is not None:
            try:
                head_np.setHpr(0.0, 0.0, pitch)
            except Exception:
                pass

        # Right arm pitch follows view pitch
        shoulder_r = parts.get("shoulder_r")
        if shoulder_r is not None:
            try:
                shoulder_r.setHpr(0.0, 0.0, pitch)
            except Exception:
                pass

    def _apply_robot_team_tint(self, model_np: NodePath, team: int):
        """Apply team tint to the robot model's tintable materials only.

        Materials expected in the glTF: MatBodyNeutral, Mat_TeamTint, Mat_WheelRubber, Mat_Emissive.
        Team tinting:
          - Mat_TeamTint: multiply with team color via ColorScale
          - Mat_DarkTeamTint: multiply with team color via ColorScale
          - Mat_Emissive: set Material emission color to team color
        """
        team_color = self._get_flag_color(team)
        # Traverse geoms and apply per-material rules
        for np in model_np.findAllMatches("**/+GeomNode"):
            gnode = np.node()
            for i in range(gnode.getNumGeoms()):
                state = gnode.getGeomState(i)
                try:
                    ma = state.getAttrib(MaterialAttrib)
                except Exception:
                    ma = None
                if not ma:
                    continue
                try:
                    mat = ma.getMaterial()
                    mname = mat.getName() if mat is not None else ""
                except Exception:
                    mname = ""
                base_name = mname.split('.')[0] if mname else ""
                if base_name == "Mat_TeamTint":
                    new_state = state.addAttrib(ColorScaleAttrib.make(team_color))
                    gnode.setGeomState(i, new_state)
                if base_name == "Mat_DarkTeamTint":
                    new_state = state.addAttrib(ColorScaleAttrib.make(team_color))
                    gnode.setGeomState(i, new_state)
                elif base_name == "Mat_Emissive":
                    # Override emission color on the material
                    try:
                        r = float(team_color[0]); g = float(team_color[1]); b = float(team_color[2])
                    except Exception:
                        r, g, b = float(team_color.x), float(team_color.y), float(team_color.z)
                    try:
                        mat2 = mat.makeCopy() if mat is not None else Material()
                    except Exception:
                        mat2 = Material()
                    try:
                        mat2.setEmission(LColor(r, g, b, 1.0))
                    except Exception:
                        pass
                    try:
                        mat2.setName(f"{base_name}_tinted")
                    except Exception:
                        pass
                    new_state = state.setAttrib(MaterialAttrib.make(mat2))
                    gnode.setGeomState(i, new_state)

    def _render_flags(self, s0, s1, alpha):
        # Merge the two snapshots' flags into a dict by team id for interpolation
        f0 = {f.get("team"): f for f in (s0.get("flags", []) if s0 else [])}
        f1 = {f.get("team"): f for f in (s1.get("flags", []) if s1 else [])}
        teams = set(f0.keys()) | set(f1.keys())

        # Remove nodes for flags no longer present
        for team in list(self.flag_nodes.keys()):
            if team not in teams:
                np = self.flag_nodes.pop(team, None)
                if np is not None:
                    np.removeNode()

        # Create/update nodes
        for team in teams:
            a0 = f0.get(team)
            a1 = f1.get(team) if s1 is not s0 else f0.get(team)
            if a0 is None and a1 is None:
                continue

            if a0 is None:
                x = a1["x"]; y = a1["y"]; z = a1["z"]
            elif a1 is None:
                x = a0["x"]; y = a0["y"]; z = a0["z"]
            else:
                x = (1 - alpha) * a0["x"] + alpha * a1["x"]
                y = (1 - alpha) * a0["y"] + alpha * a1["y"]
                z = (1 - alpha) * a0["z"] + alpha * a1["z"]

            root = self.flag_nodes.get(team)
            if root is None:
                # Create a root so we can offset the model relative to (x,y,z)
                root = self.render.attachNewNode(f"flag-{team}")
                try:
                    scale = float(self.cfg.get("flag", {}).get("scale", 1.2))
                except Exception:
                    scale = 1.2
                root.setScale(scale)
                model_np = self.flag_template.copyTo(root)
                # Apply tint
                model_np.setColorScale(self._get_flag_color(team))
                # Do not add vertical offset: model origin is at base
                self.flag_nodes[team] = root

            # Position
            root.setPos(x, y, z)

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

    # legacy helpers replaced by HudFeed

    def update_task(self, task):
        # advance smoothed render_time toward latest snapshot time
        target = self.latest_server_time - self.interp_delay + self.interp_predict
        dt = ClockObject.getGlobalClock().getDt()
        # If we're far behind (e.g., after joining or a hiccup), snap to target
        if self.render_time == 0.0 or (target - self.render_time) > 0.5:
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
            # Do not render the local player's own model
            if my_pid is not None and pid == my_pid:
                continue
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
                # Create a container so we can drop in the robot model (or fallback box)
                node = self.render.attachNewNode(f"player-{pid}")
                pr = float(self.cfg.get("gameplay", {}).get("player_radius", 0.5))
                ph = float(self.cfg.get("gameplay", {}).get("player_height", 2.0))
                team = (p1 or p0)["team"]
                col = (1.0, 0.5, 0.1, 1) if team == TEAM_RED else (0.1, 0.5, 1.0, 1)

                if self.robot_template is not None:
                    try:
                        model = self.robot_template.copyTo(node)
                        # No rotation or scaling; just shift Z so the model's center aligns with the node origin.
                        # With origin at floor center, offset by -half height.
                        if self._robot_bounds is not None:
                            min_v, max_v = self._robot_bounds
                            cz = 0.5 * (min_v.z + max_v.z)
                            if abs(cz) > 1e-6:
                                model.setZ(-cz)
                        else:
                            # Fallback: use configured player height to approximate half-height offset
                            model.setZ(-0.5 * ph)
                        # Apply team tint to specific materials on the copied model
                        try:
                            self._apply_robot_team_tint(model, team)
                        except Exception as e:
                            print(f"[players] tinting materials failed: {e}")
                        # Cache references to articulated parts by name for later animation
                        def _part(np, name):
                            try:
                                p = np.find(f"**/{name}")
                                return None if p.isEmpty() else p
                            except Exception:
                                return None
                        parts = {
                            "root": node,
                            "model": model,
                            "base": _part(model, "Base"),
                            "body": _part(model, "Body"),
                            "head": _part(model, "Head"),
                            "shoulder_r": _part(model, "Shoulder.R"),
                            "blaster": _part(model, "Blaster"),
                        }
                        self.robot_parts[pid] = parts
                    except Exception as e:
                        print(f"[players] Robot copy failed: {e}. Using box fallback.")
                        model = None
                else:
                    model = None

                if model is None:
                    # Fallback simple box scaled to radius/height
                    box = self.loader.loadModel("models/box.egg")
                    box.setScale(2.0 * pr, 2.0 * pr, ph)
                    # Center about origin (box.egg is unit cube at +/-0.5, so this keeps origin centered already)
                    box.reparentTo(node)

                # For the fallback box, tint the whole node so it's readable
                if model is None:
                    node.setColor(*col)

                self.player_nodes[pid] = node

                # attach cosmetics (headgear, nameplate, teammate caret)
                name = (p1 or p0).get("name", f"PID{pid}")
                self.cosmetics.attach(pid, node, name, team)

            node.setPos(px, py, pz)
            # Animate articulated parts if present; else rotate whole node
            parts = self.robot_parts.get(pid)
            if parts and any(parts.get(k) for k in ("base","body","head","shoulder_r")):
                # Rotate the whole robot to match view yaw (root yaw)
                node.setHpr(yaw, 0.0, 0.0)
                self._animate_robot_parts(pid, parts, yaw, pitch, roll, p0, p1, s0, s1, a, px, py)
            else:
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
                # clear part caches
                if pid in self.robot_parts:
                    del self.robot_parts[pid]
                if pid in self._prev_pos:
                    del self._prev_pos[pid]
                if pid in self._last_base_h:
                    del self._last_base_h[pid]

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
                # Interpolated position for local player
                if me0 is None:
                    px, py, pz = me1["x"], me1["y"], me1["z"]
                elif me1 is None:
                    px, py, pz = me0["x"], me0["y"], me0["z"]
                else:
                    px = (1 - a) * me0["x"] + a * me1["x"]
                    py = (1 - a) * me0["y"] + a * me1["y"]
                    pz = (1 - a) * me0["z"] + a * me1["z"]
                cx, cy, cz = px, py, pz

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

                # Ensure and update hidden local robot for anchor-based visuals
                if self.robot_template is not None:
                    try:
                        if self._local_robot_root is None:
                            # Create hidden root and copy model
                            root = self.render.attachNewNode("local-robot-hidden")
                            root.hide()
                            model = self.robot_template.copyTo(root)
                            # Z-offset like remote players
                            if self._robot_bounds is not None:
                                min_v, max_v = self._robot_bounds
                                czm = 0.5 * (min_v.z + max_v.z)
                                if abs(czm) > 1e-6:
                                    model.setZ(-czm)
                            else:
                                ph = float(self.cfg.get("gameplay", {}).get("player_height", 2.0))
                                model.setZ(-0.5 * ph)
                            # Tint to team
                            try:
                                team_local = self.local_team if self.local_team is not None else self.client.team
                                self._apply_robot_team_tint(model, team_local)
                            except Exception:
                                pass
                            # Cache parts
                            def _part(np, name):
                                try:
                                    p = np.find(f"**/{name}")
                                    return None if p.isEmpty() else p
                                except Exception:
                                    return None
                            parts = {
                                "root": root,
                                "model": model,
                                "base": _part(model, "Base"),
                                "body": _part(model, "Body"),
                                "head": _part(model, "Head"),
                                "shoulder_r": _part(model, "Shoulder.R"),
                                "blaster": _part(model, "Blaster"),
                            }
                            self.robot_parts[my_pid] = parts
                            self._local_robot_root = root
                        # Update pose for hidden robot
                        parts = self.robot_parts.get(my_pid)
                        if parts is not None:
                            root = parts.get("root")
                            if root is not None:
                                root.setPos(px, py, pz)
                                root.setHpr(self.yaw, 0.0, 0.0)
                                # Animate subparts using snapshots and current view
                                self._animate_robot_parts(my_pid, parts, self.yaw, self.pitch, 0.0, me0, me1, s0, s1, a, px, py)
                    except Exception:
                        pass

        # draw recent beams from the current snapshots
        self._render_beams(s0, s1, a)
        # draw flags (neutral/team)
        self._render_flags(s0, s1, a)

        # update cosmetics (distance fade + billboards)
        if hasattr(self, "cosmetics"):
            self.cosmetics.update()

        # --- Killfeed ingest & prune ---
        latest = s1 or s0
        if latest:
            # map pid -> team for coloring names
            pid_team = {p.get("pid"): p.get("team") for p in latest.get("players", [])}
            feed = latest.get("killfeed", [])
            for ev in feed:
                key = (int(float(ev.get("t", 0.0)) * 1000), ev.get("attacker"), ev.get("victim"))
                if key in self._kill_seen:
                    continue
                self._kill_seen.add(key)
                cause = ev.get('cause')
                att = ev.get('attacker_name','?')
                vic = ev.get('victim_name','?')
                att_t = pid_team.get(ev.get('attacker'))
                vic_t = pid_team.get(ev.get('victim'))
                if cause == 'grenade':
                    parts = [(att, att_t), " [grenade] ", (vic, vic_t)]
                else:
                    parts = [(att, att_t), " eliminated ", (vic, vic_t)]
                # normalize to (text, team) tuples and add
                norm = [(p, None) if isinstance(p, str) else p for p in parts]
                self.kill_feed.add_parts(norm)

            # Message feed
            msgs = latest.get("messages", [])
            for ev in msgs:
                key = (int(float(ev.get("t", 0.0)) * 1000), ev.get("actor"), ev.get("event"))
                if key in self._msg_seen:
                    continue
                self._msg_seen.add(key)
                name = ev.get('actor_name', '?')
                team = pid_team.get(ev.get('actor'))
                evt = ev.get('event')
                if evt == 'pickup':
                    parts = [(name, team), " picked up the flag"]
                elif evt == 'drop':
                    parts = [(name, team), " dropped the flag"]
                elif evt == 'capture':
                    parts = [(name, team), " captured the flag!"]
                else:
                    parts = [(name, team), f" {evt}"]
                norm = [(p, None) if isinstance(p, str) else p for p in parts]
                self.msg_feed.add_parts(norm)

        # expire old killfeed entries (no fade, simple TTL)
        nowt = time.time()
        # Prune feeds
        self.kill_feed.prune(nowt)
        self.msg_feed.prune(nowt)

        if latest and my_pid is not None:
            me_latest = next((p for p in latest.get("players", [] ) if p["pid"] == my_pid), None)
            if me_latest:
                self.shots_left = int(me_latest.get("shots", self.shots_left))
                reload_left = float(me_latest.get("reload", 0.0))
                if reload_left > 0:
                    self.reload_end = self.render_time + reload_left
                if self.render_time >= self.reload_end and self.shots_left == 0:
                    self.shots_left = self.shots_per_mag

        if "tab" in self.keys:
            self.scoreboard.show()
            self.scoreboard.update(self.client.state)
        else:
            self.scoreboard.hide()

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
        invert = self._invert_y
        self.yaw += -dx
        self.pitch += (-dy if invert else dy)
        self.pitch = max(-90.0, min(90.0, self.pitch))

        # apply camera orientation (HPR). Position is handled above via interpolation.
        # Note: recoil no longer moves the camera; it only affects aim offsets + crosshair.
        self.camera.setHpr(self.yaw, self.pitch, 0)

        # Firing/reloading state used to control recoil reset/decay
        fire_pressed = self.mouseWatcherNode.is_button_down(MouseButton.one())
        reloading_now = self.render_time < self.reload_end

        # Smoothly decay offsets back to center when not actively holding fire (exponential by Hz)
        actively_shooting = fire_pressed and not reloading_now and self.shots_left > 0
        if not actively_shooting and self.recoil_decay_hz > 0.0:
            dt_decay = max(0.0, ClockObject.getGlobalClock().getDt())
            factor = math.exp(-self.recoil_decay_hz * dt_decay)
            self.aim_pitch_offset *= factor
            self.aim_yaw_offset *= factor

        # Move crosshair to visualize aim offsets
        try:
            lens = self.camLens
            fov = lens.getFov()  # Vec2(hfov, vfov) in degrees
            hfov = math.radians(float(fov[0]))
            vfov = math.radians(float(fov[1]))
            # Map angular offset to normalized screen offset (fractions of half-screen)
            x_norm = math.tan(math.radians(self.aim_yaw_offset)) / max(1e-6, math.tan(0.5 * hfov))
            y_norm = math.tan(math.radians(self.aim_pitch_offset)) / max(1e-6, math.tan(0.5 * vfov))
            # Convert to aspect2d coordinates: horizontal half-range is aspect, vertical is 1
            aspect = self.getAspectRatio()
            x = max(-aspect, min(aspect, x_norm * aspect))
            y = max(-1.0, min(1.0, y_norm))
            self.crosshair.setPos(x, y)
        except Exception:
            # Fallback: small linear offset if lens data is unavailable
            self.crosshair.setPos(0.01 * self.aim_yaw_offset, 0.01 * self.aim_pitch_offset)

        data = {
            "mx": mx,
            "mz": mz,
            "jump": "space" in self.keys,
            "crouch": ("control" in self.keys) if not self._toggle_crouch else bool(getattr(self, "_crouch_toggle_state", False)),
            "walk": "shift" in self.keys,
            "fire": False,
            "interact": "e" in self.keys,
            "yaw": self.yaw,
            "pitch": self.pitch,
        }

        min_dt = 1.0 / max(1e-6, self.rapid_fire_rate)
        can_fire = fire_pressed and self.render_time >= self.reload_end and self.shots_left > 0 and (self.render_time - self.last_local_fire) >= min_dt
        if can_fire:
            self._apply_recoil()
            data["fire"] = True
            data["fire_t"] = self.render_time
            # Apply aim offsets only to the shot direction (not view orientation)
            data["yaw"] = self.yaw + self.aim_yaw_offset
            data["pitch"] = self.pitch + self.aim_pitch_offset
            self.shots_left -= 1
            self.last_local_fire = self.render_time
            if self.shots_left == 0:
                self.reload_end = self.render_time + self.reload_seconds
        else:
            data["fire"] = False
        self.ammo_text.setText(str(self.shots_left))

        # Right mouse for grenade throw
        if self.mouseWatcherNode.is_button_down(MouseButton.three()):
            if self._grenade_hold_start is None:
                self._grenade_hold_start = self.render_time
        elif self._grenade_hold_start is not None:
            hold = max(0.0, self.render_time - self._grenade_hold_start)
            data["grenade"] = hold
            self._grenade_hold_start = None

        if self.client.writer:
            self.net_runner.run_coro(self.client.send_input(data))

        # Update edge trackers for next frame
        self.prev_fire_pressed = fire_pressed
        self.was_reloading = reloading_now

        # Update debug netgraph overlay
        if self._netgraph_enabled and self._netgraph is not None:
            try:
                ping = 0
                if self.client and self.client.state and self.client.pid is not None:
                    me = next((p for p in self.client.state.get("players", []) if p.get("pid") == self.client.pid), None)
                    ping = int(me.get("ping", 0)) if me else 0
                snaps = len(self.snapshots)
                self._netgraph.setText(f"ping: {ping} ms\nsnaps: {snaps}\nrt: {self.render_time:.2f}")
            except Exception:
                pass
        return task.cont

def _load_client_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_client_state(path: str, state: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[settings] failed to save client state: {e}")


def pick_server(settings: Dict[str, Any], shared_cfg: Dict[str, Any]):
    net = settings.get("network", {}) if settings else {}
    srv = net.get("server", {})
    host = srv.get("host")
    port = int(srv.get("port", shared_cfg.get("server", {}).get("port", 50007)))
    disc = net.get("discovery", {})
    discovery_enabled = bool(disc.get("enabled", True))

    # If host is explicit (not "auto"), use it directly
    if host and isinstance(host, str) and host != "auto":
        return host, port

    # If remember_last_server is enabled and we have one saved, use it
    prefs = settings.get("server_prefs", {}) if settings else {}
    if bool(prefs.get("remember_last_server", True)):
        state = _load_client_state(os.path.join("configs", "client_state.json"))
        last = state.get("last_server")
        if last and isinstance(last, dict) and last.get("host") and last.get("port"):
            return last.get("host"), int(last.get("port"))

    if not discovery_enabled:
        return None, None

    # Otherwise, discover on LAN with preferences and optional retry
    lan_port = int(shared_cfg.get("server", {}).get("lan_discovery_port", 50000))
    preferred = disc.get("preferred_server_names", []) or []
    retry_ms = int(disc.get("retry_ms", 1000))
    timeout_s = max(0.2, retry_ms / 1000.0)

    loop = asyncio.get_event_loop()
    tried = 0
    while True:
        servers = loop.run_until_complete(lan_discovery_broadcast(lan_port, timeout=timeout_s))
        pick = None
        if servers:
            # Choose by preference list order first
            if preferred:
                for pref in preferred:
                    for s in servers:
                        if str(s.get("name", "")) == str(pref):
                            pick = s
                            break
                    if pick is not None:
                        break
            # Fallback: first
            if pick is None:
                pick = servers[0]
        if pick is not None:
            return pick.get("addr"), int(pick.get("tcp_port", port))
        tried += 1
        # If auto-join is disabled, stop after first attempt
        if not bool(prefs.get("auto_join_on_match", True)):
            break
        print("[discovery] no servers yet; retrying...")
        time.sleep(timeout_s)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--settings", default="configs/client_settings.json")
    ap.add_argument("--name", default=None)
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
    settings = load_settings(args.settings)

    # Apply PRC (window, vsync, audio device) before engine init
    apply_prc_from_settings(settings)

    # Name and team preference
    prof = settings.get("profile", {}) if settings else {}
    profile_name = args.name if args.name is not None else prof.get("name", "Player")
    team_pref = str(prof.get("team_preference", "auto")) if prof else "auto"

    # Choose server
    host, port = args.host, args.port
    if host is None or host == "":
        host2, port2 = pick_server(settings, cfg)
        host = host2 or host
        port = port2 or port
        if not host:
            print("No server available. Provide --host/--port or enable discovery.")
            sys.exit(1)
    if not port:
        port = cfg["server"]["port"]

    # Interp delay: CLI overrides settings; only use settings if CLI default used
    if args.interp_delay == 0.10:
        try:
            args.interp_delay = float(settings.get("network", {}).get("interp_delay", args.interp_delay))
        except Exception:
            pass

    app = GameApp(cfg, host=host, port=port, name=profile_name, interp_delay=args.interp_delay, interp_predict=args.interp_predict, settings=settings, team_pref=team_pref)

    # FPS meter and cap
    try:
        vid = settings.get("video", {}) if settings else {}
        if bool(vid.get("show_fps", False)):
            app.setFrameRateMeter(True)
        if "max_fps" in vid:
            fps = float(vid.get("max_fps", 0))
            if fps and fps > 0:
                gc = ClockObject.getGlobalClock()
                gc.setMode(ClockObject.MLimited)
                gc.setFrameRate(fps)
    except Exception:
        pass

    # Apply audio volumes (master/effects/music)
    try:
        aud = settings.get("audio", {}) if settings else {}
        muted = bool(aud.get("muted", False))
        master = float(aud.get("master", 1.0)) if not muted else 0.0
        music_v = float(aud.get("music", master)) if not muted else 0.0
        sfx_v = float(aud.get("effects", master)) if not muted else 0.0
        try:
            base.musicManager.setVolume(max(0.0, min(1.0, music_v)))
        except Exception:
            pass
        try:
            for mgr in getattr(base, "sfxManagerList", []) or []:
                try:
                    mgr.setVolume(max(0.0, min(1.0, sfx_v)))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    # Save last server for convenience
    try:
        prefs = settings.get("server_prefs", {}) if settings else {}
        if bool(prefs.get("remember_last_server", True)) and host and port:
            state = _load_client_state(os.path.join("configs", "client_state.json"))
            state["last_server"] = {"host": host, "port": int(port)}
            _save_client_state(os.path.join("configs", "client_state.json"), state)
    except Exception:
        pass

    app.run()


if __name__ == "__main__":
    main()
