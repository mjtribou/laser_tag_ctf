"""ECS systems executed by the authoritative server loop."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from panda3d.core import LPoint3, Vec3, NodePath
from panda3d.bullet import BulletWorld

from game.transform import deg_to_rad, forward_vector, local_move_delta, wrap_pi

from .components import (
    CharacterBody,
    Health,
    MovementState,
    Physics,
    PlayerInfo,
    PlayerInput,
    Position,
    Projectile,
    Weapon,
)
from .world import System, World


@dataclass
class HitEvent:
    attacker_entity: int
    victim_entity: int
    hit_point: Optional[Tuple[float, float, float]]
    shot_dir: Optional[Tuple[float, float, float]]
    cause: str = "beam"


@dataclass
class GrenadeRequest:
    entity: int
    owner_pid: int
    power: float


class MovementSystem(System):
    """Translates input intent into Bullet character controller commands."""

    def __init__(self, world: World, gameplay_cfg: Dict[str, float]) -> None:
        super().__init__(world)
        self.gameplay_cfg = gameplay_cfg

    def _movement_speed(self, crouch: bool, walk: bool) -> float:
        if crouch:
            return float(self.gameplay_cfg.get("crouch_speed", 2.5))
        if walk:
            return float(self.gameplay_cfg.get("walk_speed", 4.5))
        return float(self.gameplay_cfg.get("run_speed", 7.0))

    def update(self, dt: float) -> None:
        accel_ground = float(self.gameplay_cfg.get("acceleration_mps2", 20.0))
        accel_air = float(self.gameplay_cfg.get("air_acceleration_mps2", max(5.0, 0.5 * accel_ground)))
        decel_release_g = float(self.gameplay_cfg.get("decel_release_mps2", 0.5 * accel_ground))
        decel_release_a = float(self.gameplay_cfg.get("air_decel_release_mps2", 0.5 * accel_air))
        decel_counter_g = float(self.gameplay_cfg.get("decel_counter_mps2", 1.6 * accel_ground))
        decel_counter_a = float(self.gameplay_cfg.get("air_decel_counter_mps2", 1.2 * accel_air))

        for entity, (pos, phys, inputs, body, health) in self.world.query(
            Position, Physics, PlayerInput, CharacterBody, Health
        ):
            if not health.alive:
                # Dead characters should not receive intentional movement.
                try:
                    body.controller.setLinearMovement(Vec3(0, 0, 0), False)
                except Exception:
                    pass
                continue

            # Persist stance state for networking
            move_state = self.world.get_component(entity, MovementState)
            if move_state:
                move_state.walking = bool(inputs.walk)
                move_state.crouching = bool(inputs.crouch)

            # Update facing from client supplied degrees
            pos.yaw = wrap_pi(deg_to_rad(float(inputs.yaw)))
            pos.pitch = wrap_pi(deg_to_rad(float(inputs.pitch)))

            speed = self._movement_speed(bool(inputs.crouch), bool(inputs.walk))
            mx = float(inputs.mx)
            mz = float(inputs.mz)

            controller = body.controller
            nodepath = body.nodepath
            if controller is None or nodepath is None:
                continue

            # Determine grounded state prior to issuing movement so we can select acceleration curves
            try:
                on_ground = bool(controller.isOnGround())
            except Exception:
                on_ground = False
            phys.on_ground = on_ground

            # Desired horizontal velocity from intent
            vdx, vdy = local_move_delta(mx, mz, pos.yaw, speed, 1.0)

            cur_vx, cur_vy = phys.vx, phys.vy
            accel_limit = accel_ground if on_ground else accel_air

            intent_mag = min(1.0, math.hypot(mx, mz))
            cur_spd = math.hypot(cur_vx, cur_vy)
            if intent_mag <= 1e-6:
                accel_limit = decel_release_g if on_ground else decel_release_a
            else:
                dot = cur_vx * vdx + cur_vy * vdy
                if cur_spd > 1e-6 and dot < 0.0:
                    accel_limit = decel_counter_g if on_ground else decel_counter_a

            max_dv = max(0.0, accel_limit * max(0.0, dt))
            dvx, dvy = (vdx - cur_vx), (vdy - cur_vy)
            dvmag = math.hypot(dvx, dvy)
            if dvmag > 1e-6 and dvmag > max_dv:
                scale = max_dv / dvmag
                dvx *= scale
                dvy *= scale
            cmd_vx = cur_vx + dvx
            cmd_vy = cur_vy + dvy

            cmd_spd = math.hypot(cmd_vx, cmd_vy)
            if cmd_spd > speed and cmd_spd > 1e-6:
                k = speed / cmd_spd
                cmd_vx *= k
                cmd_vy *= k

            try:
                controller.setLinearMovement(Vec3(cmd_vx, cmd_vy, 0.0), False)
            except Exception:
                continue

            if inputs.jump and on_ground:
                try:
                    controller.doJump()
                except Exception:
                    pass

    def late_update(self, dt: float) -> None:
        if dt <= 0:
            return
        for entity, (pos, phys, body) in self.world.query(Position, Physics, CharacterBody):
            nodepath = body.nodepath
            if nodepath is None:
                continue
            prev_x, prev_y, prev_z = pos.x, pos.y, pos.z
            pos.x = float(nodepath.getX())
            pos.y = float(nodepath.getY())
            pos.z = float(nodepath.getZ())
            phys.vx = (pos.x - prev_x) / dt
            phys.vy = (pos.y - prev_y) / dt
            phys.vz = (pos.z - prev_z) / dt
            try:
                phys.on_ground = bool(body.controller.isOnGround())
            except Exception:
                pass


class CombatSystem(System):
    """Handles weapon firing, recoil, and combat resolution."""

    def __init__(
        self,
        world: World,
        gameplay_cfg: Dict[str, float],
        server_cfg: Dict[str, float],
        bullet_world: BulletWorld,
        now_fn = time.time,
    ) -> None:
        super().__init__(world)
        self.gameplay_cfg = gameplay_cfg
        self.server_cfg = server_cfg
        self.bullet_world = bullet_world
        self._now = now_fn
        self.respawn_seconds = float(server_cfg.get("respawn_seconds", 5.0))
        self.beam_events: List[Dict[str, float]] = []
        self.hit_events: List[HitEvent] = []
        self.grenade_requests: List[GrenadeRequest] = []

    def update(self, dt: float) -> None:
        self.beam_events.clear()
        self.hit_events.clear()
        self.grenade_requests.clear()

        rof = float(self.gameplay_cfg.get("rapid_fire_rate_hz", 8.0))
        min_dt = 1.0 / max(1e-6, rof)
        reload_sec = float(self.gameplay_cfg.get("reload_seconds", 1.5))
        recoil_per_shot = float(self.gameplay_cfg.get("recoil_per_shot_deg", 0.18))
        decay_hz = float(self.gameplay_cfg.get("recoil_decay_hz", 7.5))
        move_factor = float(self.gameplay_cfg.get("spread_move_factor", 2.0))
        crouch_bonus = float(self.gameplay_cfg.get("spread_crouch_bonus", -1.0))
        base_spread = float(self.gameplay_cfg.get("base_spread_deg", 1.0))
        friendly_fire = bool(self.gameplay_cfg.get("friendly_fire", False))
        player_radius = float(self.gameplay_cfg.get("player_radius", 0.5))
        player_height = float(self.gameplay_cfg.get("player_height", 1.8))
        max_range = float(self.gameplay_cfg.get("laser_range_m", 80.0))

        now_t = self._now()

        for entity, comps in self.world.query(
            Position, Physics, PlayerInput, Weapon, Health, PlayerInfo
        ):
            pos, phys, inputs, weapon, health, info = comps

            if weapon.shots_remaining <= 0 and now_t >= weapon.reload_end:
                weapon.shots_remaining = weapon.shots_per_mag

            if not health.alive:
                if inputs.grenade_charge and inputs.grenade_charge > 0.0:
                    inputs.grenade_charge = 0.0
                # Decay recoil even while dead so it resets when respawning.
                if weapon.recoil_accum > 0.0 and decay_hz > 0.0:
                    weapon.recoil_accum *= math.exp(-decay_hz * max(0.0, dt))
                    if weapon.recoil_accum < 1e-4:
                        weapon.recoil_accum = 0.0
                continue

            if inputs.grenade_charge and inputs.grenade_charge > 0.0:
                self.grenade_requests.append(GrenadeRequest(entity, info.pid, float(inputs.grenade_charge)))
                inputs.grenade_charge = 0.0

            if weapon.recoil_accum > 0.0 and decay_hz > 0.0:
                weapon.recoil_accum *= math.exp(-decay_hz * max(0.0, dt))
                if weapon.recoil_accum < 1e-4:
                    weapon.recoil_accum = 0.0

            intent_mag = min(1.0, math.hypot(inputs.mx, inputs.mz))
            crouch = bool(inputs.crouch)
            spread_deg = base_spread + move_factor * intent_mag + (crouch_bonus if crouch else 0.0) + weapon.recoil_accum
            weapon.spread_deg = spread_deg

            fire_pressed = bool(inputs.fire)
            fire_time = float(inputs.fire_timestamp or now_t)

            if fire_pressed and weapon.shots_remaining > 0 and now_t >= weapon.reload_end:
                if now_t - weapon.last_fire_time >= min_dt:
                    weapon.last_fire_time = now_t
                    weapon.shots_remaining = max(0, weapon.shots_remaining - 1)

                    spread_rad = math.radians(max(0.0, spread_deg))
                    hit_entity, hit_point, direction = self._apply_hitscan(
                        entity,
                        pos,
                        spread_rad,
                        fire_time,
                        max_range,
                        player_height,
                        player_radius,
                        friendly_fire,
                    )

                    weapon.recoil_accum = min(4.0, weapon.recoil_accum + recoil_per_shot)
                    if weapon.shots_remaining == 0:
                        weapon.reload_end = now_t + reload_sec

                    if hit_entity is not None:
                        victim_health = self.world.get_component(hit_entity, Health)
                        if victim_health and victim_health.alive:
                            victim_health.alive = False
                            victim_health.hp = 0
                            victim_health.respawn_at = now_t + self.respawn_seconds
                        self.hit_events.append(
                            HitEvent(
                                attacker_entity=entity,
                                victim_entity=hit_entity,
                                hit_point=hit_point,
                                shot_dir=direction,
                                cause="beam",
                            )
                        )

        # Late reload check (e.g. if mag emptied this frame)
        now_t = self._now()
        for _, weapon in self.world.components_of_type(Weapon).items():
            if weapon.shots_remaining <= 0 and now_t >= weapon.reload_end:
                weapon.shots_remaining = weapon.shots_per_mag

    def _apply_hitscan(
        self,
        shooter_entity: int,
        pos: Position,
        spread_rad: float,
        fire_time: float,
        max_range: float,
        player_height: float,
        player_radius: float,
        friendly_fire: bool,
    ) -> Tuple[Optional[int], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        # Shooter eye position
        sx, sy, sz = pos.x, pos.y, pos.z + 0.30 * player_height
        fx, fy, fz = forward_vector(pos.yaw, pos.pitch)

        if spread_rad > 1e-6:
            # Build orthonormal basis around forward vector
            rx, ry, rz = (-fy, fx, 0.0)
            rlen = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
            rx, ry, rz = rx / rlen, ry / rlen, 0.0
            ux, uy, uz = (ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx)
            ulen = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
            ux, uy, uz = ux / ulen, uy / ulen, uz / ulen
            a = random.uniform(0.0, 2.0 * math.pi)
            r = (random.uniform(0.0, 1.0) ** 0.5) * spread_rad
            ca, sa, cr, sr = math.cos(a), math.sin(a), math.cos(r), math.sin(r)
            fx, fy, fz = (
                fx * cr + sr * (rx * ca + ux * sa),
                fy * cr + sr * (ry * ca + uy * sa),
                fz * cr + sr * (rz * ca + uz * sa),
            )

        ex, ey, ez = sx + fx * max_range, sy + fy * max_range, sz + fz * max_range
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        dlen2 = dx * dx + dy * dy + dz * dz
        if dlen2 <= 1e-12:
            self._register_beam(shooter_entity, (sx, sy, sz), (ex, ey, ez))
            return None, None, (fx, fy, fz)

        t_wall: Optional[float] = None
        wall_point: Optional[Tuple[float, float, float]] = None
        eps_hit = 1e-4
        try:
            res = self.bullet_world.rayTestAll(LPoint3(sx, sy, sz), LPoint3(ex, ey, ez))
            for hit in res.getHits():
                np_hit = NodePath(hit.getNode())
                if np_hit.is_empty():
                    continue
                if np_hit.getPythonTag("kind") != "static":
                    continue
                frac = float(hit.getHitFraction())
                if frac <= eps_hit:
                    continue
                if t_wall is None or frac < t_wall:
                    t_wall = frac
                    wall_point = (sx + dx * frac, sy + dy * frac, sz + dz * frac)
        except Exception:
            pass

        # Axis aligned bounding box intersection against all other alive players
        shooter_info = self.world.get_component(shooter_entity, PlayerInfo)
        if shooter_info is None:
            shooter_team = None
        else:
            shooter_team = shooter_info.team

        hx = max(0.0, player_radius)
        hy = max(0.0, player_radius)
        hz = max(0.0, 0.5 * player_height)

        def hit_t_for_aabb(cx: float, cy: float, cz: float) -> Optional[float]:
            tmin, tmax = 0.0, 1.0
            for S, D, C, H in ((sx, dx, cx, hx), (sy, dy, cy, hy), (sz, dz, cz, hz)):
                if abs(D) < 1e-8:
                    if abs(S - C) > H:
                        return None
                    continue
                invD = 1.0 / D
                t1 = ((C - H) - S) * invD
                t2 = ((C + H) - S) * invD
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
                if tmax < tmin:
                    return None
            if tmax < 0.0 or tmin > 1.0:
                return None
            return max(tmin, 0.0)

        victim_entity: Optional[int] = None
        t_player: Optional[float] = None
        for other_entity, (other_pos, other_health, other_info) in self.world.query(Position, Health, PlayerInfo):
            if other_entity == shooter_entity:
                continue
            if not other_health.alive:
                continue
            if (not friendly_fire) and shooter_team is not None and other_info.team == shooter_team:
                continue
            t = hit_t_for_aabb(other_pos.x, other_pos.y, other_pos.z)
            if t is not None and (t_player is None or t < t_player):
                t_player = t
                victim_entity = other_entity

        if (victim_entity is not None) and ((t_wall is None) or (t_player is not None and t_player <= t_wall - eps_hit)):
            hit_point = (sx + dx * t_player, sy + dy * t_player, sz + dz * t_player)
            beam_end = hit_point
        else:
            hit_point = None
            beam_end = wall_point if wall_point is not None else (ex, ey, ez)
            victim_entity = None

        self._register_beam(shooter_entity, (sx, sy, sz), beam_end)

        dlen = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        return victim_entity, hit_point, (fx / dlen, fy / dlen, fz / dlen)

    def _register_beam(
        self,
        shooter_entity: int,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> None:
        shooter_info = self.world.get_component(shooter_entity, PlayerInfo)
        if shooter_info is None:
            team = -1
            pid = None
        else:
            team = shooter_info.team
            pid = shooter_info.pid

        sx, sy, sz = start
        ex, ey, ez = end
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.beam_events.append(
            {
                "team": team,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "ex": ex,
                "ey": ey,
                "ez": ez,
                "len": length,
                "t": self._now(),
                "owner": pid,
            }
        )


class CollisionSystem(System):
    """Lightweight post-physics corrections and cleanup."""

    def __init__(self, world: World, gameplay_cfg: Dict[str, float]) -> None:
        super().__init__(world)
        self.gameplay_cfg = gameplay_cfg

    def late_update(self, dt: float) -> None:
        radius = float(self.gameplay_cfg.get("player_radius", 0.5))
        min_d = 2.0 * radius * 0.98
        alive_entities: List[Tuple[int, Position, CharacterBody]] = []
        for entity, (pos, health, body) in self.world.query(Position, Health, CharacterBody):
            if not health.alive:
                continue
            if body.nodepath is None:
                continue
            alive_entities.append((entity, pos, body))

        for i in range(len(alive_entities)):
            for j in range(i + 1, len(alive_entities)):
                ea, posa, body_a = alive_entities[i]
                eb, posb, body_b = alive_entities[j]
                dx = posb.x - posa.x
                dy = posb.y - posa.y
                d2 = dx * dx + dy * dy
                if d2 < (min_d * min_d) and d2 > 1e-6:
                    d = math.sqrt(d2)
                    push = 0.5 * (min_d - d)
                    ux, uy = dx / d, dy / d
                    npa = body_a.nodepath
                    npb = body_b.nodepath
                    npa.setPos(npa.getX() - ux * push, npa.getY() - uy * push, npa.getZ())
                    npb.setPos(npb.getX() + ux * push, npb.getY() + uy * push, npb.getZ())
                    posa.x = float(npa.getX())
                    posa.y = float(npa.getY())
                    posa.z = float(npa.getZ())
                    posb.x = float(npb.getX())
                    posb.y = float(npb.getY())
                    posb.z = float(npb.getZ())

        # Cull expired projectiles
        now_t = time.time()
        ttl_map = list(self.world.components_of_type(Projectile).items())
        for entity, projectile in ttl_map:
            if projectile.ttl > 0.0 and now_t - projectile.spawn_time >= projectile.ttl:
                self.world.remove_entity(entity)
