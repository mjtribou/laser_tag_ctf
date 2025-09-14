# game/transform.py
import math

# ---- Angles ---------------------------------------------------------------

def wrap_pi(a: float) -> float:
    """Wrap radians into [-pi, pi]."""
    return ((a + math.pi) % (2 * math.pi)) - math.pi

def deg_to_rad(d: float) -> float:
    return math.radians(d)

def rad_to_deg(r: float) -> float:
    return math.degrees(r)

# ---- Panda3D camera/game-space (X right, Y forward, Z up) ----------------
# H (yaw) rotates around +Z; H=0 faces +Y; +H rotates left (CCW).
# P (pitch) rotates around +X; +P looks up.

def heading_forward_xy(yaw_rad: float) -> tuple[float, float]:
    """Forward unit vector in the XY plane for Panda H."""
    sh, ch = math.sin(yaw_rad), math.cos(yaw_rad)
    return (-sh, ch)

def heading_right_xy(yaw_rad: float) -> tuple[float, float]:
    """Right unit vector in the XY plane for Panda H."""
    sh, ch = math.sin(yaw_rad), math.cos(yaw_rad)
    return (ch, sh)

def local_move_delta(mx: float, mz: float, yaw_rad: float, speed: float, dt: float) -> tuple[float, float]:
    """
    Convert local intent (mx=strafe, mz=forward) to world-space XY delta.
    - mx: strafe left/right intent in [-1, 1]
    - mz: forward/back intent in [-1, 1]
    - speed: top speed (m/s)
    - dt: time step (s)

    Diagonal-normalized: if sqrt(mx^2 + mz^2) > 1, the (mx, mz) vector is
    scaled to unit length so diagonal movement does not exceed top speed.
    """
    # Normalize local intent to length <= 1 to avoid diagonal speed boost
    mag = math.hypot(mx, mz)
    if mag > 1.0 and mag > 1e-8:
        inv = 1.0 / mag
        mx *= inv
        mz *= inv

    rx, ry = heading_right_xy(yaw_rad)
    fx, fy = heading_forward_xy(yaw_rad)
    dx = (fx * mz + rx * mx) * speed * dt
    dy = (fy * mz + ry * mx) * speed * dt
    return dx, dy

def forward_vector(yaw_rad: float, pitch_rad: float) -> tuple[float, float, float]:
    """
    3D forward with pitch in Panda frame:
      X = -sin(H) * cos(P)
      Y =  cos(H) * cos(P)
      Z =  sin(P)
    """
    sh, ch = math.sin(yaw_rad), math.cos(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    return (-sh * cp, ch * cp, sp)
