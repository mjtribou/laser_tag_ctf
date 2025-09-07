# client.py
import sys, asyncio, json, time, math, argparse, threading, asyncio
from typing import Dict, Any
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3, Point3, DirectionalLight, AmbientLight, LVector3f, KeyboardButton, WindowProperties
from panda3d.core import LColor, MouseButton
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletBoxShape
from common.net import send_json, read_json, lan_discovery_broadcast
from game.constants import TEAM_RED, TEAM_BLUE
from game.map_gen import generate

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
        await send_json(self.writer, {"type":"hello","name":self.name})
        welcome = await read_json(self.reader)
        self.pid = welcome["pid"]; self.team = welcome["team"]
        print(f"[net] connected. pid={self.pid} team={self.team}")

    async def recv_state_loop(self, on_state):
        while True:
            msg = await read_json(self.reader)
            if not msg:
                break
            if msg.get("type")=="state":
                self.state = msg
                on_state(msg)

    async def send_input(self, data: Dict[str, Any]):
        self.last_input = data
        await send_json(self.writer, {"type":"input","data":data})

class GameApp(ShowBase):
    def __init__(self, cfg, host: str, port: int, name: str):
        ShowBase.__init__(self)
        self.set_background_color(0.05, 0.05, 0.07, 1)
        self.disableMouse()

        self.cfg = cfg
        self.client = NetworkClient(cfg, name=name)
        self.host, self.port = host, port
        self.yaw, self.pitch = 0.0, 0.0
        self.latest_state = None

        # lighting
        dlight = DirectionalLight('dlight'); dlight.setColor(LColor(0.9,0.9,0.9,1))
        dlnp = self.render.attachNewNode(dlight); dlnp.setHpr(45, -60, 0); self.render.setLight(dlnp)
        alight = AmbientLight('alight'); alight.setColor(LColor(0.2,0.2,0.25,1))
        alnp = self.render.attachNewNode(alight); self.render.setLight(alnp)

        # world (visual & simple bullet for occlusion later)
        self.world = BulletWorld(); self.world.setGravity(Vec3(0,0,-9.81))

        # arena
        size_x, size_z = cfg["gameplay"]["arena_size_m"]
        self.mapdata = generate(seed=42, size_x=size_x, size_z=size_z)
        # floor plane (visual only)
        cm = self.render.attachNewNode("floor")
        floor = self.loader.loadModel("models/box"); floor.setTwoSided(True)
        floor.setScale(size_x, 1.0, size_z)
        floor.setPos(0,0,-0.1)
        floor.setColor(0.2,0.2,0.22,1)
        floor.reparentTo(cm)

        # obstacles
        for b in self.mapdata.blocks:
            model = self.loader.loadModel("models/box"); model.setTwoSided(True)
            model.setScale(b.size[0], b.size[2], b.size[1])
            model.setPos(b.pos[0], b.pos[2], b.pos[1])
            # color tint by side
            tint = (0.6,0.6,0.6,1)
            model.setColor(*tint)
            model.reparentTo(self.render)

        # bases / beacons
        self.red_beacon = self.loader.loadModel("models/box"); self.red_beacon.setScale(2,2,6); self.red_beacon.setTwoSided(True)
        self.red_beacon.setPos(self.mapdata.red_base[0], self.mapdata.red_base[2], 3)
        self.red_beacon.setColor(1.0,0.5,0.1,1); self.red_beacon.reparentTo(self.render)
        self.blue_beacon = self.loader.loadModel("models/box"); self.blue_beacon.setScale(2,2,6); self.blue_beacon.setTwoSided(True)
        self.blue_beacon.setPos(self.mapdata.blue_base[0], self.mapdata.blue_base[2], 3)
        self.blue_beacon.setColor(0.1,0.5,1.0,1); self.blue_beacon.reparentTo(self.render)

        # player representations
        self.player_nodes = {}  # pid -> NodePath

        # camera init
        self.camera.setPos(0, -15, 3)
        self.mouse_locked = False
        self.center_mouse()

        # UI text
        from direct.gui.OnscreenText import OnscreenText
        self.crosshair = OnscreenText(text="+", pos=(0,0), fg=(1,1,1,1), scale=0.08, mayChange=True)

        # key state
        self.keys = set()
        for key in ["w","a","s","d","space","control","shift","e","tab","m"]:
            self.accept(key, self.on_key, [key, True])
            self.accept(key+"-up", self.on_key, [key, False])
        self.accept("escape", sys.exit)

        # mouse update
        self.taskMgr.add(self.update_task, "update")

        # network start
        self.net_runner = AsyncRunner()
        self.net_runner.run_coro(self.client.connect(host, port))
        self.net_runner.run_coro(self.client.recv_state_loop(self.on_state))

    def center_mouse(self):
        wp = WindowProperties(); wp.setCursorHidden(True)
        self.win.requestProperties(wp)
        self.mouse_locked = True

    def on_state(self, state):
        # Thread-safe enough under the GIL; main thread will consume it
        self.latest_state = state
        self.client.state = state

    def on_key(self, key, down):
        if down:
            self.keys.add(key)
        else:
            self.keys.discard(key)

    def poll_mouse(self):
        if not self.mouseWatcherNode.hasMouse():
            return 0.0, 0.0
        m = self.mouseWatcherNode.getMouse()
        # convert to deltas by recentering each frame
        x = m.getX(); y = m.getY()
        sens = self.cfg["controls"]["mouse_sensitivity"]
        dx, dy = x * sens * 100.0, y * sens * 100.0
        self.win.movePointer(0, int(self.win.getXSize()/2), int(self.win.getYSize()/2))
        return dx, dy

    def update_task(self, task):
        # === Apply latest server snapshot on the render thread ===
        state = self.latest_state
        if state:
            present = set()
            for p in state["players"]:
                pid = p["pid"]
                present.add(pid)
                node = self.player_nodes.get(pid)
                if node is None:
                    node = self.loader.loadModel("models/box.egg")
                    node.setScale(0.6, 0.6, 1.8)
                    node.setTwoSided(True)
                    col = (1.0, 0.5, 0.1, 1) if p["team"] == TEAM_RED else (0.1, 0.5, 1.0, 1)
                    node.setColor(*col)
                    node.reparentTo(self.render)
                    self.player_nodes[pid] = node
                node = self.player_nodes[pid]
                node.setPos(p["x"], p["z"], p["y"])
                node.setHpr(-p["yaw"], 0, 0)

            # remove nodes for players no longer present
            for pid in list(self.player_nodes.keys()):
                if pid not in present:
                    self.player_nodes[pid].removeNode()
                    del self.player_nodes[pid]

            # snap the camera to our authoritative position
            my_pid = self.client.pid
            if my_pid:
                me = next((pp for pp in state["players"] if pp["pid"] == my_pid), None)
                if me:
                    self.camera.setPos(me["x"], me["z"] - 0.001, me["y"] + 0.7)

        # build inputs
        mx = 0.0; mz = 0.0
        if "w" in self.keys: mz += 1
        if "s" in self.keys: mz -= 1
        if "a" in self.keys: mx -= 1
        if "d" in self.keys: mx += 1
        # mouse
        dx, dy = self.poll_mouse()

        # update our own angles
        invert = bool(self.cfg["controls"].get("invert_y", False))
        self.yaw   += -dx
        self.pitch += (-dy if invert else dy)   # up = look up when invert_y is False
        self.pitch = max(-90.0, min(90.0, self.pitch))

        # apply to camera
        self.camera.setHpr(self.yaw, self.pitch, 0)

        data = {
            "mx": mx, "mz": mz,
            "jump": "space" in self.keys,
            "crouch": "control" in self.keys,
            "walk": "shift" in self.keys,
            "fire": False,
            "interact": "e" in self.keys,
            "yaw": self.yaw, "pitch": self.pitch,
        }

        # fire if left mouse button (not handled; use E for MVP firing) -> Use 'e' to interact; left-mouse mapping in Panda3D is "mouse1"
        if self.mouseWatcherNode.is_button_down(MouseButton.one()):
            data["fire"] = True

        # async send
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

    app = GameApp(cfg, host=host, port=port, name=args.name)
    app.run()

if __name__ == "__main__":
    main()
