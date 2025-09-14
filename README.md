# Laser Tag CTF (Panda3D + Bullet, Authoritative Server)

> First playable LAN prototype for 5v5 Capture‑the‑Flag laser tag with hitscan shots, recoil/spread, crouch accuracy, bots, and UDP LAN discovery.  
> **Server‑authoritative 60 Hz** simulation, **20 Hz** snapshots to clients, JSON‑over‑TCP protocol.

## Features implemented (MVP)
- Panda3D renderer, Bullet world scaffold
- Authoritative server (`server.py`) runs game loop, flags (pickup/drop/auto‑return), captures (win at 3), 3s respawn
- Default game mode: single center‑flag CTF (neutral flag spawns at arena center; either team can capture by returning it to their base)
- Hitscan lasers with server‑side validation, recoil accumulation, accuracy penalties while moving; crouch improves accuracy
- Basic 5v5 via **bot fill** (dumb patrol/engage behavior) + human players
- Symmetric **procedural block arena** (~80×50m) with team‑colored base beacons
- **LAN discovery** via UDP broadcast (port 50000); clients can also join by IP
- UI: crosshair + team‑colored players; Tab scoreboard and richer HUD are stubs for v2
- Configurable **mouse sensitivity**, **volume/mute**, **keymap**, and rule constants via `configs/defaults.json`
- **No footsteps while walking/sneaking** logic is scaffolded (audio hooks to be added).

## Install
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If Panda3D wheels fail, install system prerequisites per your OS. Windows/MSVC and Linux manylinux wheels are published on PyPI.

## Run
### 1) Start the server
```bash
python server.py --config configs/defaults.json
```
- Default TCP port: **50007**
- LAN discovery UDP port: **50000**
- Server automatically fills with bots up to 5v5 (configurable).

### 2) Start a client (auto‑discover LAN server)
```bash
python client.py --name "Mike"
```
Or join manually:
```bash
python client.py --host 192.168.1.50 --port 50007 --name "Mike"
```

You can also drive client preferences via a separate settings file (loaded by default at `configs/client_settings.json`):
```bash
python client.py --settings configs/client_settings.json
```
Supported fields include profile (name/team_preference), controls (mouse_sensitivity/invert_y/fov/toggle_crouch), network discovery preferences, video (fullscreen/resolution/vsync/max_fps/render_distance/show_fps/hud_scale), audio (master/effects/music/device), cosmetics (nameplates/teammate_marker), server_prefs (auto_join_on_match/remember_last_server), debug (netgraph), and privacy flags.

## Controls (defaults)
- **WASD** move, **Space** jump (visual), **Ctrl** crouch, **Shift** walk (sneak; no footsteps), **E** interact (pickup/capture)
- **Mouse** look; **Left click** fire
- **Tab** scoreboard (stub)

All keybinds & mouse sensitivity are in `configs/defaults.json` → `"controls"`.

## Notes & roadmap
- MVP uses simplified collision & occlusion; **server hit‑tests players only** (no wall occlusion yet). We’ll add Bullet ray tests vs static colliders next.
- Client‑side prediction is minimal (camera reconciliation). For v2: interpolate remote players, predict local movement + server reconciliation.
- Audio hooks exist in config; next iteration will add SFX (tag, flag events) via Panda3D audio.
- Scoreboard/HUD, team markers, FOV recoil kick, and colorblind palettes are partially implemented; polishing next.

## Project layout
```
laser_tag_ctf/
  client.py            # Panda3D client + input/HUD + LAN discovery
  server.py            # Authoritative server, bots, flags, CTF logic
  common/net.py        # JSON-over-TCP & UDP LAN discovery helpers
  game/
    constants.py
    map_gen.py
    server_state.py    # shared math helpers & state structs
    bot_ai.py
  configs/defaults.json
  requirements.txt
  README.md
```

## Known limitations (to be addressed)
- No wall occlusion for hitscan yet (the server uses sphere hit on players only).
- Physics: player movement uses simple integration; Bullet character controller to be added for capsule vs static boxes.
- Bots: simple waypointing/steering; planned grid A* and better targeting/flag logic.
- UI/HUD: full scoreboard, flag icons, teammate markers pending.
- Colorblind palette: currently **Blue vs Orange**; further palette tweaks and UI shapes planned.
