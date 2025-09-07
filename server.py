# server.py
import asyncio, json, math, time, random
from typing import Dict, Any, List, Tuple
from common.net import read_json, send_json, lan_discovery_server
from game.constants import TEAM_RED, TEAM_BLUE, MAX_PLAYERS, PLAYER_HEIGHT, PLAYER_RADIUS, FLAG_PICKUP_RADIUS, FLAG_RETURN_RADIUS, BASE_CAPTURE_RADIUS
from game.map_gen import generate
from game.server_state import GameState, Player, Flag, unit_vector_from_angles
from game.bot_ai import SimpleBotBrain

import argparse, pathlib

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

class LaserTagServer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.gs = GameState()
        self.next_pid = 1
        size_x, size_z = cfg["gameplay"]["arena_size_m"]
        self.mapdata = generate(seed=42, size_x=size_x, size_z=size_z)

        # init flags
        red_flag = Flag(team=TEAM_RED, at_base=True, carried_by=None,
                        x=self.mapdata.red_flag_stand[0], y=0.0, z=self.mapdata.red_flag_stand[2])
        blue_flag = Flag(team=TEAM_BLUE, at_base=True, carried_by=None,
                        x=self.mapdata.blue_flag_stand[0], y=0.0, z=self.mapdata.blue_flag_stand[2])
        self.gs.flags = {TEAM_RED: red_flag, TEAM_BLUE: blue_flag}

        self.clients: Dict[int, asyncio.StreamWriter] = {}
        self.inputs: Dict[int, Dict[str, Any]] = {}  # last input per pid
        self.bot_brains: Dict[int, SimpleBotBrain] = {}
        self._last_snapshot = 0.0
        self._last_input_log: Dict[int, float] = {}

    def assign_spawn(self, team: int) -> Tuple[float,float,float,float]:
        # spawn near base with some jitter
        base = self.mapdata.red_base if team == TEAM_RED else self.mapdata.blue_base
        x = base[0] + random.uniform(-3,3)
        z = base[2] + random.uniform(-3,3)
        yaw = 0.0 if team == TEAM_RED else 180.0
        return x, 0.9, z, yaw

    def add_player(self, name: str, is_bot=False) -> int:
        # pick team with fewer members
        red_count = sum(1 for p in self.gs.players.values() if p.team==TEAM_RED)
        blue_count = sum(1 for p in self.gs.players.values() if p.team==TEAM_BLUE)
        team = TEAM_RED if red_count<=blue_count else TEAM_BLUE
        x,y,z,yaw = self.assign_spawn(team)
        pid = self.next_pid; self.next_pid += 1
        self.gs.players[pid] = Player(pid=pid, name=name, team=team, x=x,y=y,z=z, yaw=yaw)
        if is_bot:
            base = self.mapdata.red_base if team==TEAM_RED else self.mapdata.blue_base
            enemy = self.mapdata.blue_base if team==TEAM_RED else self.mapdata.red_base
            self.bot_brains[pid] = SimpleBotBrain(team, base, enemy)
        return pid

    def remove_player(self, pid: int):
        if pid in self.gs.players:
            # if carrying flag, drop it
            p = self.gs.players[pid]
            if p.carrying_flag is not None:
                flg = self.gs.flags[p.carrying_flag]
                flg.carried_by = None
                flg.at_base = False
                flg.x, flg.y, flg.z = p.x, 0.0, p.z
                flg.dropped_at_time = time.time()
                p.carrying_flag = None
            del self.gs.players[pid]
        self.clients.pop(pid, None)
        self.inputs.pop(pid, None)
        self.bot_brains.pop(pid, None)

    def process_input(self, p: Player, inp: Dict[str, Any], dt: float):
        if not p.alive:
            return
        gp = self.cfg["gameplay"]
        speed = gp["run_speed"]
        if inp.get("walk"):
            speed = gp["walk_speed"]
            p.walking = True
        else:
            p.walking = False
        if inp.get("crouch"):
            p.crouching = True
            speed = min(speed, gp["crouch_speed"])
        else:
            p.crouching = False

        mx = float(inp.get("mx", 0.0))
        mz = float(inp.get("mz", 0.0))
        # move in local yaw
        yaw_rad = math.radians(p.yaw)
        fwdx, fwdz = math.cos(yaw_rad), -math.sin(yaw_rad)
        leftx, leftz = -fwdz, fwdx
        dx = (fwdx*mz + leftx*mx)*speed*dt
        dz = (fwdz*mz + leftz*mx)*speed*dt

        # simple floor clamp
        p.x += dx
        p.z += dz
        p.y = 0.9  # keep simple

        # look
        p.yaw = float(inp.get("yaw", p.yaw))
        p.pitch = max(-90.0, min(90.0, float(inp.get("pitch", p.pitch))))
        
        # --- DEBUG: log player inputs/state (throttled to 1 Hz per player)
        now = time.time()
        last = self._last_input_log.get(p.pid, 0.0)
        if now - last >= 1.0:
            print(
                f"[input] pid={p.pid} name={p.name} "
                f"team={'RED' if p.team==TEAM_RED else 'BLUE'} "
                f"mx={float(inp.get('mx', 0.0)):+.1f} mz={float(inp.get('mz', 0.0)):+.1f} "
                f"walk={bool(inp.get('walk', False))} crouch={bool(inp.get('crouch', False))} "
                f"fire={bool(inp.get('fire', False))} "
                f"yaw={p.yaw:.1f} pitch={p.pitch:.1f} "
                f"pos=({p.x:.2f},{p.z:.2f}) alive={p.alive}"
            )
            self._last_input_log[p.pid] = now


        now = time.time()
        # firing
        if inp.get("fire"):
            rof = gp["rapid_fire_rate_hz"]
            if (now - p.last_fire_time) >= (1.0/rof):
                p.last_fire_time = now
                # recoil accum & spread
                p.recoil_accum += gp["recoil_per_shot_deg"]
                # accuracy based on movement speed + crouch
                move_mag = math.sqrt(dx*dx + dz*dz)/max(dt,1e-5)
                move_factor = (move_mag / gp["run_speed"])
                spread = gp["base_spread_deg"] + move_factor*gp["spread_move_factor"]
                if p.crouching:
                    spread += gp["spread_crouch_bonus"]
                spread = max(0.02, spread) + p.recoil_accum*0.2
                self.apply_hitscan(p, spread_deg=spread)

        # decay recoil
        p.recoil_accum = max(0.0, p.recoil_accum - self.cfg["gameplay"]["recoil_decay_per_sec_deg"]*dt)

        # interact (pick flag)
        if inp.get("interact"):
            self.try_flag_interact(p)

        # respawn if dead and time elapsed
        if (not p.alive) and time.time() >= p.respawn_at:
            self.spawn_player(p)

    def spawn_player(self, p: Player):
        base = self.mapdata.red_base if p.team==TEAM_RED else self.mapdata.blue_base
        p.x, p.y, p.z, p.yaw = self.assign_spawn(p.team)
        p.alive = True
        p.carrying_flag = None
        p.recoil_accum = 0.0

    def apply_hitscan(self, shooter: Player, spread_deg: float):
        # random spread around yaw/pitch
        yaw = shooter.yaw + random.uniform(-spread_deg, spread_deg)
        pitch = shooter.pitch + random.uniform(-spread_deg, spread_deg)
        dx,dy,dz = unit_vector_from_angles(yaw, pitch)
        max_range = self.cfg["gameplay"]["laser_range_m"]
        # simple line test against players (no occlusion for MVP)
        best_pid = None
        best_t = 1e9
        sx, sy, sz = shooter.x, shooter.y+0.8, shooter.z
        for pid, target in self.gs.players.items():
            if pid == shooter.pid or target.team == shooter.team:
                continue
            if not target.alive:
                continue
            # sphere intersection (approximate)
            tx, ty, tz = target.x, target.y+0.9, target.z
            # compute closest approach along ray
            rx, ry, rz = tx - sx, ty - sy, tz - sz
            t = (rx*dx + ry*dy + rz*dz)
            if t < 0 or t > max_range:
                continue
            closest_x = sx + dx*t
            closest_y = sy + dy*t
            closest_z = sz + dz*t
            dist2 = (closest_x-tx)**2 + (closest_y-ty)**2 + (closest_z-tz)**2
            if dist2 <= 0.4*0.4:  # hit radius
                if t < best_t:
                    best_t = t
                    best_pid = pid
        if best_pid is not None:
            self.handle_tag(shooter, self.gs.players[best_pid])

    def handle_tag(self, shooter: Player, victim: Player):
        # drop victim flag if any
        if victim.carrying_flag is not None:
            flg = self.gs.flags[victim.carrying_flag]
            flg.carried_by = None
            flg.at_base = False
            flg.x, flg.y, flg.z = victim.x, 0.0, victim.z
            flg.dropped_at_time = time.time()
            victim.carrying_flag = None
        victim.alive = False
        victim.respawn_at = time.time() + self.cfg["server"]["respawn_seconds"]

    def try_flag_interact(self, p: Player):
        # pick up enemy flag if at stand or dropped nearby
        enemy_team = TEAM_RED if p.team==TEAM_BLUE else TEAM_BLUE
        flag = self.gs.flags[enemy_team]
        if flag.carried_by is None:
            # compute distance
            dx = (flag.x if not flag.at_base else (self.mapdata.red_flag_stand[0] if enemy_team==TEAM_RED else self.mapdata.blue_flag_stand[0])) - p.x
            dz = (flag.z if not flag.at_base else (self.mapdata.red_flag_stand[2] if enemy_team==TEAM_RED else self.mapdata.blue_flag_stand[2])) - p.z
            if (dx*dx + dz*dz) <= FLAG_PICKUP_RADIUS*FLAG_PICKUP_RADIUS and p.alive:
                flag.carried_by = p.pid
                flag.at_base = False
        # attempt capture if carrying enemy flag and at home base with own flag present
        if p.carrying_flag is not None or (self.gs.flags[enemy_team].carried_by == p.pid):
            my_flag = self.gs.flags[p.team]
            # determine base position
            bx = self.mapdata.red_flag_stand[0] if p.team==TEAM_RED else self.mapdata.blue_flag_stand[0]
            bz = self.mapdata.red_flag_stand[2] if p.team==TEAM_RED else self.mapdata.blue_flag_stand[2]
            dx = bx - p.x
            dz = bz - p.z
            if (dx*dx + dz*dz) <= BASE_CAPTURE_RADIUS*BASE_CAPTURE_RADIUS:
                # must have own flag at base
                if my_flag.at_base and my_flag.carried_by is None:
                    # score!
                    self.gs.teams[p.team].captures += 1
                    # return enemy flag
                    eflag = self.gs.flags[TEAM_RED if p.team==TEAM_BLUE else TEAM_BLUE]
                    eflag.at_base = True
                    eflag.carried_by = None
                    # remove from player if we tracked that
                    for pl in self.gs.players.values():
                        if pl.pid == p.pid:
                            pl.carrying_flag = None
                    # check win
                    if self.gs.teams[p.team].captures >= self.cfg["server"]["captures_to_win"]:
                        self.gs.match_over = True
                        self.gs.winner = p.team

    def update_flags(self):
        # update flag positions + auto return logic
        for t, flg in self.gs.flags.items():
            if flg.carried_by is not None:
                if flg.carried_by in self.gs.players:
                    pl = self.gs.players[flg.carried_by]
                    flg.x, flg.y, flg.z = pl.x, 0.0, pl.z
                else:
                    # carrier left
                    flg.carried_by = None
            if (not flg.at_base) and (flg.carried_by is None):
                # auto return after timeout
                if time.time() - flg.dropped_at_time >= self.cfg["server"]["flag_return_seconds"]:
                    base = self.mapdata.red_flag_stand if t==TEAM_RED else self.mapdata.blue_flag_stand
                    flg.x, flg.y, flg.z = base[0], 0.0, base[2]
                    flg.at_base = True

    def build_snapshot(self):
        # compact snapshot
        return {
            "type":"state",
            "time": time.time(),
            "teams": {str(TEAM_RED): {"captures": self.gs.teams[TEAM_RED].captures},
                      str(TEAM_BLUE): {"captures": self.gs.teams[TEAM_BLUE].captures}},
            "flags": {str(t): {"at_base": f.at_base, "carried_by": f.carried_by, "x": f.x, "y": f.y, "z": f.z} for t,f in self.gs.flags.items()},
            "players": [
                {"pid":p.pid, "name":p.name, "team":p.team, "x":p.x, "y":p.y, "z":p.z, "yaw":p.yaw, "pitch":p.pitch,
                 "alive":p.alive, "carrying_flag":p.carrying_flag}
                for p in self.gs.players.values()
            ],
            "over": self.gs.match_over, "winner": self.gs.winner
        }

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        hello = await read_json(reader)
        if hello.get("type") != "hello":
            writer.close(); return
        name = hello.get("name","Player")
        pid = self.add_player(name=name, is_bot=False)
        self.clients[pid] = writer

        await send_json(writer, {"type":"welcome","pid":pid,"team":self.gs.players[pid].team})
        print(f"[join] {name} -> pid {pid} team {self.gs.players[pid].team}")

        try:
            while True:
                msg = await read_json(reader)
                if not msg:
                    break
                if msg.get("type") == "input":
                    self.inputs[pid] = msg["data"]
                elif msg.get("type") == "pickup":
                    self.try_flag_interact(self.gs.players[pid])
        except Exception as e:
            print("Client error:", e)
        finally:
            print(f"[leave] pid {pid}")
            self.remove_player(pid)
            writer.close()

    async def run(self):
        # Fill with bots to 10 players
        if self.cfg["server"]["bot_fill"]:
            while len(self.gs.players) < self.cfg["server"]["bot_per_team_target"]*2:
                self.add_player(name=f"Bot{len(self.gs.players)+1}", is_bot=True)

        # Start TCP server
        server = await asyncio.start_server(self.handle_client, host="0.0.0.0", port=self.cfg["server"]["port"])
        print(f"[server] listening on 0.0.0.0:{self.cfg['server']['port']}")

        # Start LAN discovery responder
        asyncio.create_task(lan_discovery_server(self.cfg["server"]["name"], self.cfg["server"]["port"], self.cfg["server"]["lan_discovery_port"]))

        tick_dt = 1.0/self.cfg["server"]["tick_hz"]
        snap_dt = 1.0/self.cfg["server"]["snapshot_hz"]
        last_snap = time.time()

        async with server:
            while True:
                t0 = time.time()
                # apply inputs + bots
                for pid, p in list(self.gs.players.items()):
                    if pid in self.bot_brains:
                        ai_inp = self.bot_brains[pid].decide(p, self.gs, self.mapdata)
                        self.process_input(p, ai_inp, tick_dt)
                    else:
                        inp = self.inputs.get(pid, {})
                        self.process_input(p, inp, tick_dt)

                self.update_flags()

                # broadcast snapshot
                if (t0 - last_snap) >= snap_dt:
                    snap = self.build_snapshot()
                    for pid, w in list(self.clients.items()):
                        try:
                            await send_json(w, snap)
                        except Exception:
                            pass
                    last_snap = t0

                # sleep to maintain tick
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, tick_dt - elapsed))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/defaults.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    srv = LaserTagServer(cfg)
    try:
        asyncio.run(srv.run())
    except KeyboardInterrupt:
        print("Server shutting down.")

if __name__ == "__main__":
    main()
