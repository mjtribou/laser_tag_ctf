# Laser Tag CTF

Authoritative, server-driven capture-the-flag arena built on Panda3D + Bullet. Supports LAN play, bot fill, and a lightweight ECS architecture for deterministic gameplay.

## Highlights
- **Authoritative server loop** at 60 Hz with JSON-over-TCP clients and UDP LAN discovery.
- **Entity-component systems** (`game/ecs`) drive player input, movement, combat, collisions, and flag state; legacy views expose the same API to existing client/bot code.
- **Hitscan combat** with Bullet ray occlusion, recoil/spread tuning, killfeed, ragdoll corpses, and beam replication.
- **Authored map pipeline** (JSON + CSG) feeding chunk meshes and merged colliders; single neutral-flag CTF out of the box.
- **Bot fill and HUD**: A* navigation bots keep lobbies full, live scoreboard (Tab) shows team captures, ping, tags, outs, captures, defences.
- **Config driven**: tweak gameplay, physics, visuals, controls, audio, and server limits through `configs/defaults.json` (server) and `configs/client_settings.json` (client).

## Getting Started
### Requirements
- Python 3.10+
- Panda3D, Bullet, and dependencies listed in `requirements.txt`

### Install
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
If Panda3D wheels fail, install system prerequisites for your OS (see Panda3D documentation).

### Run the server
```bash
python server.py --config configs/defaults.json
```
- Defaults: TCP `50007`, LAN discovery UDP `50000`
- Server auto-fills teams to 5v5 using bots (see `server.bot_fill`, `bot_per_team_target`).

### Run a client
```bash
python client.py --name "Player"
```
To join a specific host:
```bash
python client.py --host 192.168.1.50 --port 50007 --name "Player"
```
Additional preferences (FOV, HUD scale, audio, keybinds, etc.) can be set in `configs/client_settings.json`. The client will remember the last server in `configs/client_state.json`.

## Controls (default)
- `WASD` move, `Space` jump, `Ctrl` crouch, `Shift` walk, `E` interact
- `Mouse` to aim, `LMB` fire, hold for continuous shots
- `Tab` toggles the scoreboard, `M` mute

Mouse sensitivity, invert-Y, and similar controls tweaks live under `controls` in the config.

## Configuration Overview
Key sections in `configs/defaults.json`:
- `server`: ports, tick/snapshot rates, bot behaviour, victory conditions
- `gameplay`: movement speeds, acceleration, spread/recoil, grenade tuning, respawn timers
- `map`: select the arena layout JSON under `configs/maps/`
- `gameplay.character_collider`: choose `aabb` or `capsule` for player physics bodies
- `colors`, `hud`, `audio`, `cosmetics`: team tinting, killfeed TTL, footsteps, nameplates
- `ragdoll`, `laser_visual`: tweak knockback, beam rendering

### Map authoring
Map JSON files now support constructive solid geometry style operations alongside simple axis-aligned blocks. Each entry inside `blocks` can be either a classic block with `pos`, `size`, and `box_type`, or an operation node with an `op` and nested `children`/`shapes`. Supported operations are `union`, `intersect`, and `subtract` (aliases: `add`, `merge`, `difference`, `minus`, `and`). Operations inherit the parent `box_type` by default, but any node can override it.

Example snippet:
```json
{
  "op": "subtract",
  "box_type": 2,
  "children": [
    { "pos": [0, 0, 1], "size": [20, 12, 2] },
    { "pos": [0, 0, 1], "size": [6, 6, 2] }
  ]
}
```
The above carves a square hole out of a larger platform while keeping the result voxel-aligned to `cube_size`.

Tactical navigation metadata can be embedded under an optional `nav` block:

```json
"nav": {
  "nodes": [
    { "id": "mid_cover", "pos": [0, 12, 0], "type": "cover", "tags": ["mid", "defend"], "facing": [0, -1, 0], "radius": 1.8 }
  ],
  "links": [
    { "from": "mid_cover", "to": "enemy_ramp", "weight": 1.2 }
  ]
}
```
Nodes describe reusable tactical anchor points (cover, perch, flank, regroup) with optional tags, facing vectors, and influence radius; links specify directed or bidirectional connections that future bot logic can traverse.

### Bot debug overlay
Set `BOT_DEBUG=1` when launching the client (e.g. `BOT_DEBUG=1 python client.py`) to enable a live bot decision overlay. Press `F9` in-game to toggle the panel; it shows each bot’s current high-level behaviour, score, and target for quick debugging.

Clients mirror many of these options in `configs/client_settings.json` (video, audio, HUD, cosmetics, discovery preferences).

## Architecture Notes
- `server.py` bootstraps Panda3D/Bullet, constructs the ECS world, spawns flags & players as entities, and runs pre/post physics systems each frame.
- `game/ecs/` contains component definitions, the minimal ECS world, movement/combat/collision systems, snapshot serialization, and view facades for legacy code.
- `common/net.py` handles LAN discovery (UDP broadcast) and TCP JSON streams.
- `world/map_adapter.py` converts authored maps into voxel grids for chunking and physics.
- `game/bot_ai.py` supplies Simple and A* brains; the server updates them at 10 Hz and feeds decisions into `PlayerInput` components.
- `client.py` renders the arena, handles input, interpolation, HUD (including `scoreboard.py`), and applies server snapshots.

## Project Layout
```
laser_tag_ctf/
  client.py              # Panda3D client, input/HUD, networking
  server.py              # Authoritative server, ECS systems, bots, flags
  scoreboard.py          # Tab scoreboard overlay used by client
  common/net.py          # TCP JSON protocol + LAN discovery support
  configs/               # Server & client configuration presets/state
  game/
    ecs/                 # ECS components, systems, replication helpers
    bot_ai.py            # Simple & A* bot brains
    map_gen.py           # Map data schema + JSON helpers
    constants.py         # Shared gameplay constants/enums
    transform.py         # Math helpers (angles, movement deltas)
    nav_grid.py          # Grid-building & pathfinding helpers for bots
  tools/                 # Development scripts/utilities
  requirements.txt
  README.md
```

## Feature Status & Roadmap
- ✅ Bullet occlusion for hitscan, recoil decay, ragdoll knockback, grenade AoE tags
- ✅ ECS-driven movement/combat/collision; snapshots sourced directly from components
- ✅ Live scoreboard, killfeed, message feed in client HUD
- 🚧 Client-side prediction is limited (camera only)—full reconciliation still on the roadmap
- 🚧 Authentication/persistence is absent; server trusts LAN clients
- 🚧 Audio set includes placeholders; richer SFX/music still planned
- 🚧 Additional game modes (multi-flag, king-of-the-hill) can be layered by adding systems/components

## Troubleshooting
- **No server found via LAN**: ensure UDP broadcast is allowed on port 50000 or join via `--host`.
- **Physics jitter**: verify `tick_hz` and `snapshot_hz` align with client frame timing; reducing beam/fuse speeds can help on low-end hardware.
- **Panda3D import errors**: reinstall Panda3D with `pip install panda3d` after activating the virtualenv, or install platform-specific prerequisites.

Happy tagging!
