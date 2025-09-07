# game/bot_ai.py
import math, random, time
from typing import Dict, Tuple, List
from .constants import TEAM_RED, TEAM_BLUE

class SimpleBotBrain:
    """
    Extremely simple bot AI:
    - Patrol between a few waypoints near mid and their base.
    - If enemy in forward cone and roughly visible, move toward/engage (fire).
    - If carrying enemy flag, move toward home.
    - If see enemy flag on ground nearby, try to pick it up.
    """
    def __init__(self, team: int, base_pos: Tuple[float,float,float], enemy_base: Tuple[float,float,float]):
        self.team = team
        self.base_pos = base_pos
        self.enemy_base = enemy_base
        self.state = "patrol"
        self.target = None
        self.last_repath = 0.0

    def decide(self, me, gs, mapdata):
        now = time.time()
        inputs = {"mx":0.0,"mz":0.0,"jump":False,"crouch":False,"walk":False,"fire":False,"interact":False,"yaw":me.yaw,"pitch":me.pitch}
        # priorities
        goal = None
        if me.carrying_flag is not None:
            goal = self.base_pos
        else:
            # biased toward enemy base / mid
            if self.state == "patrol" or self.target is None or (now - self.last_repath) > 5.0:
                self.target = random.choice([self.enemy_base, (0,0,0), (self.enemy_base[0]*0.6,0,self.enemy_base[2]*0.6)])
                self.last_repath = now
            goal = self.target

        # point yaw toward goal
        dx = goal[0]-me.x; dz = goal[2]-me.z
        if abs(dx)+abs(dz) > 0.5:
            desired_yaw = math.degrees(math.atan2(-dz, dx))
            # turn slightly
            diff = (desired_yaw - me.yaw + 180) % 360 - 180
            me.yaw += max(-120*0.016, min(120*0.016, diff))

        # move toward goal
        inputs["mx"] = math.cos(math.radians(me.yaw))
        inputs["mz"] = -math.sin(math.radians(me.yaw))

        # very dumb fire: occasionally fire when not walking
        inputs["fire"] = random.random() < 0.05
        inputs["walk"] = False
        inputs["crouch"] = random.random() < 0.02
        return inputs
