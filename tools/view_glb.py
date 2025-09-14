#!/usr/bin/env python3
"""
Minimal GLB viewer to validate rendering of models/cube.glb with the
known-good combination: python gltf loader + simplepbr pipeline.

Usage examples:
  python tools/view_glb.py --model models/cube.glb --tile 1 --face top=3 \
                          --elev 40 --az 25

Includes:
  - UV remap into a 2x2 atlas per face (tile indices 0..3)
  - Orbiting point light to show specular highlights from the normal map
  - Camera centering/angles (elevation/azimuth)
  - Simple exposure controls (comma/period)
"""

import argparse
from typing import Dict, Tuple

from panda3d.core import loadPrcFileData

# PRC tweaks before ShowBase init
loadPrcFileData("view-glb", "basic-shaders-only #f")
loadPrcFileData("view-glb", "framebuffer-srgb true")
loadPrcFileData("view-glb", "window-title GLB Viewer")

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import DirectionalLight, AmbientLight, PointLight, LColor, LVector3


class Viewer(ShowBase):
    def __init__(self, model_path: str, elev_deg: float, az_deg: float, face_tiles: Dict[str, Tuple[int,int]]):
        super().__init__()
        self.disableMouse()
        self.set_background_color(0.05, 0.05, 0.07, 1)
        self._spin = True
        self._orbit = True
        self._pbr = None
        self._elev_deg = float(elev_deg)
        self._az_deg = float(az_deg)
        self._face_tiles = face_tiles or {}

        # Lights
        d = DirectionalLight("key"); d.setColor(LColor(0.95, 0.95, 0.95, 1))
        dn = self.render.attachNewNode(d); dn.setHpr(45, -50, 0); self.render.setLight(dn)
        f = DirectionalLight("fill"); f.setColor(LColor(0.25, 0.25, 0.30, 1))
        fn = self.render.attachNewNode(f); fn.setHpr(-60, -15, 0); self.render.setLight(fn)
        a = AmbientLight("amb"); a.setColor(LColor(0.18, 0.18, 0.22, 1))
        an = self.render.attachNewNode(a); self.render.setLight(an)

        # Pipeline: always simplepbr (known-good path with python gltf loader)
        try:
            import simplepbr
            self._pbr = simplepbr.init(use_normal_maps=True, max_lights=8)
            # Start with a slightly negative exposure so specular isn't blown out
            try:
                self._pbr.exposure = -0.5
            except Exception:
                pass
            print("[viewer] simplepbr pipeline active")
        except Exception as e:
            raise SystemExit(f"[viewer] simplepbr required but not available: {e}")

        # Load model via python gltf loader only
        try:
            import gltf
            self.model = gltf.load_model(model_path)
            print("[viewer] loaded with python gltf.load_model")
        except Exception as e:
            raise SystemExit(f"[viewer] gltf.load_model failed: {e}")

        # Ensure we have a NodePath wrapper early (before UV remap)
        try:
            from panda3d.core import NodePath
            if not hasattr(self.model, 'findAllMatches'):
                self.model = NodePath(self.model)
        except Exception:
            pass

        # Optional: remap UVs to atlas tiles per face
        if self._face_tiles:
            try:
                self._remap_uvs_to_tiles(self.model, self._face_tiles)
            except Exception as e:
                print("[viewer] UV remap failed:", e)

        # Attach and center
        # Attach
        self.model.reparentTo(self.render)

        # Compute tight bounds and re-center the model at the world origin
        b = self.model.getTightBounds()
        if b:
            from panda3d.core import Point3
            min_v, max_v = b
            center = (min_v + max_v) * 0.5
            extent = max((max_v - min_v).x, (max_v - min_v).y, (max_v - min_v).z)
            radius = max(0.001, extent * 0.5)
            print(f"[viewer] bounds center={center} radius~{radius:.3f}")

            # Create a pivot at the center so spins are around the model
            self.pivot = self.render.attachNewNode('pivot')
            self.pivot.setPos(center)
            # Reparent the model to the pivot, and offset so world center is at (0,0,0)
            self.model.wrtReparentTo(self.pivot)
            self.model.setPos(self.model, -center)
            # Place camera using spherical-like parameters (distance, elevation, azimuth)
            import math
            dist = max(3.0, radius * 3.0)
            elev = math.radians(self._elev_deg)
            az   = math.radians(self._az_deg)
            rxy = dist * math.cos(elev)
            cam_z = center.z + dist * math.sin(elev)
            cam_x = center.x + rxy * math.sin(az)
            cam_y = center.y - rxy * math.cos(az)  # negative Y to sit "behind" model
            self.camera.setPos(cam_x, cam_y, cam_z)
            self.camera.lookAt(center)
            # Initial slight turn for 3/4 view
            self.pivot.setHpr(25, 0, 0)
            # Update specular point light near the surface
            if hasattr(self, 'plnp'):
                self._orbit_radius = radius * 1.8
                self.plnp.setPos(center.x + self._orbit_radius, center.y, center.z + radius * 0.8)
        else:
            # Fallback position
            self.model.setPos(0, 0, 0)
            self.camera.setPos(0, -7, 3)
            self.camera.lookAt(0, 0, 0)

        # Add a bright point light that orbits the cube to show specular highlights
        pl = PointLight("spec")
        pl.setColor(LColor(3.0, 3.0, 3.0, 1))
        # Weak attenuation to keep it bright
        pl.setAttenuation(LVector3(0.5, 0.0, 0.0))
        self.plnp = self.render.attachNewNode(pl)
        self.render.setLight(self.plnp)
        self.plnp.setPos(2.5, 5.0, 1.2)

        # Minimal diagnostics
        print("[viewer] ----- model root -----")
        try:
            texs = [t.getName() for t in self.model.findAllTextures()]
        except Exception:
            texs = []
        print("textures:", texs)

        # UI + controls
        self._help = OnscreenText(text="[space]=toggle spin  [o]=toggle orbit  [,/]=exposure -/+",
                                   pos=(0.02, 0.95), scale=0.045, fg=(1,1,1,0.85), align=0, mayChange=True)
        self._info = OnscreenText(text=f"exposure: -0.5  elev:{self._elev_deg:.1f}° az:{self._az_deg:.1f}°",
                                   pos=(0.02, 0.90), scale=0.045,
                                   fg=(0.85,1,0.95,0.9), align=0, mayChange=True)
        self.accept('space', self._toggle_spin)
        self.accept('o', self._toggle_orbit)
        self.accept(',', self._expo_delta, [-0.25])
        self.accept('.', self._expo_delta, [ 0.25])
        self.taskMgr.add(self._update_task, "viewer-update")

    # --- controls ---
    def _toggle_spin(self):
        self._spin = not self._spin

    def _toggle_orbit(self):
        self._orbit = not self._orbit

    def _expo_delta(self, d: float):
        if self._pbr is None:
            return
        try:
            self._pbr.exposure = float(getattr(self._pbr, 'exposure', 0.0)) + d
            self._info.setText(f"exposure: {self._pbr.exposure:+.2f}")
        except Exception:
            pass

    def _update_task(self, task):
        dt = globalClock.getDt()
        t = task.time
        if self._spin:
            # Spin pivot around model center
            if hasattr(self, 'pivot'):
                self.pivot.setH(self.pivot.getH() + 25.0 * dt)
            else:
                self.model.setH(self.model.getH() + 25.0 * dt)
        if self._orbit:
            r = getattr(self, '_orbit_radius', 2.5)
            import math
            x = r * math.cos(t*1.2)
            y = r * math.sin(t*1.2)
            z = (0.8 * r) * 0.4 + (0.5 * r) * math.sin(t*1.8)
            # Orbit around origin; if pivot had set center, that's world center
            self.plnp.setPos(x, y, z)
        return task.cont

    # --- UV atlas remapping -------------------------------------------------
    def _remap_uvs_to_tiles(self, root, face_tiles: Dict[str, Tuple[int,int]], grid=(2,2)):
        """
        Remap UVs so that each cube face samples a desired tile from a 2x2 atlas.
        face_tiles: dict mapping face name -> (tx, ty) tile coords where tx,ty in {0,1}
                    faces: +x, -x, +y, -y, +z, -z  (aliases: right,left,front,back,top,bottom)
        """
        import math
        # Normalize face keys and provide aliases
        aliases = {
            'right': '+x', 'left': '-x', 'front': '+y', 'back': '-y', 'top': '+z', 'bottom': '-z'
        }
        wanted: Dict[str, Tuple[int,int]] = {}
        for k, v in face_tiles.items():
            kk = aliases.get(k.lower(), k.lower())
            wanted[kk] = v

        from panda3d.core import GeomNode, GeomVertexReader, GeomVertexWriter, GeomTriangles
        from panda3d.core import GeomVertexFormat

        # Helper to classify a triangle by its geometric normal
        def classify(p0, p1, p2) -> str:
            from panda3d.core import LVector3
            n = (p1 - p0).cross(p2 - p0)
            if n.lengthSquared() == 0:
                return 'unknown'
            n.normalize()
            ax = abs(n.x); ay = abs(n.y); az = abs(n.z)
            if ax >= ay and ax >= az:
                return '+x' if n.x > 0 else '-x'
            if ay >= ax and ay >= az:
                return '+y' if n.y > 0 else '-y'
            return '+z' if n.z > 0 else '-z'

        # UV transform per face
        tile_w = 1.0 / float(grid[0])
        tile_h = 1.0 / float(grid[1])

        count_total = 0
        count_per_face = {}

        col = root.findAllMatches('**/+GeomNode')
        for idx in range(col.getNumPaths()):
            gnode = col.getPath(idx)
            node: GeomNode = gnode.node()
            for gi in range(node.getNumGeoms()):
                geom = node.modifyGeom(gi)
                vdata = geom.modifyVertexData()
                rdr_v = GeomVertexReader(vdata, 'vertex')

                # Discover the actual UV column name present in the vertex format
                uv_name = None
                fmt: GeomVertexFormat = vdata.getFormat()
                try:
                    for ai in range(fmt.getNumArrays()):
                        arr = fmt.getArray(ai)
                        for ci in range(arr.getNumColumns()):
                            coln = arr.getColumn(ci).getName()
                            try:
                                nm = coln.getName()
                            except Exception:
                                nm = str(coln)
                            nml = nm.lower()
                            if 'texcoord' in nml:
                                uv_name = nm
                                raise StopIteration
                except StopIteration:
                    pass

                if uv_name is None:
                    print('[viewer] UV remap: no texcoord column on geom; skipping')
                    continue

                try:
                    rdr_uv = GeomVertexReader(vdata, uv_name)
                    wtr_uv = GeomVertexWriter(vdata, uv_name)
                except Exception as e:
                    print('[viewer] UV remap: could not open UV column', uv_name, e)
                    continue

                for p in range(geom.getNumPrimitives()):
                    prim = geom.modifyPrimitive(p).decompose()  # ensure triangles
                    geom.setPrimitive(p, prim)
                    nverts = prim.getNumVertices()
                    for vi in range(0, nverts, 3):
                        i0 = prim.getVertex(vi)
                        i1 = prim.getVertex(vi+1)
                        i2 = prim.getVertex(vi+2)

                        # Read positions
                        rdr_v.setRow(i0); p0 = rdr_v.getData3f()
                        rdr_v.setRow(i1); p1 = rdr_v.getData3f()
                        rdr_v.setRow(i2); p2 = rdr_v.getData3f()

                        face = classify(p0, p1, p2)
                        if face not in wanted:
                            continue
                        tx, ty = wanted[face]
                        tx = max(0, min(grid[0]-1, int(tx)))
                        ty = max(0, min(grid[1]-1, int(ty)))
                        off_u = tx * tile_w
                        off_v = ty * tile_h

                        # Remap UVs of the 3 vertices of this triangle
                        for idx in (i0, i1, i2):
                            rdr_uv.setRow(idx)
                            u, v = rdr_uv.getData2f()
                            # Keep local orientation but scale into the selected tile
                            u2 = off_u + u * tile_w
                            v2 = off_v + v * tile_h
                            wtr_uv.setRow(idx)
                            wtr_uv.setData2f(u2, v2)

                        count_total += 1
                        count_per_face[face] = count_per_face.get(face, 0) + 1

        if not count_total:
            print('[viewer] UV remap: no triangles updated (face classification may have failed)')
        else:
            print('[viewer] UV remap: updated', count_total, 'triangles; by face:', count_per_face)


def main():
    ap = argparse.ArgumentParser(description="View a glTF/GLB with simplepbr and optional UV remap")
    ap.add_argument("--model", default="models/cube.glb")
    ap.add_argument("--elev", type=float, default=40.0, help="Camera elevation in degrees")
    ap.add_argument("--az", type=float, default=25.0, help="Camera azimuth (degrees from +Y toward +X)")
    ap.add_argument('--tile', type=int, default=None, help='Uniform tile index 0..3 for all faces (2x2 atlas)')
    ap.add_argument('--face', action='append', default=[], help='Per-face mapping like right=1, top=2 or +x=1, -y=3; repeatable')
    args = ap.parse_args()

    # Build face->(tx,ty) mapping from CLI
    face_tiles: Dict[str, Tuple[int,int]] = {}
    def idx_to_pair(idx: int) -> Tuple[int,int]:
        idx = max(0, min(3, int(idx)))
        return (idx % 2, idx // 2)

    if args.tile is not None:
        pair = idx_to_pair(args.tile)
        for f in ('+x','-x','+y','-y','+z','-z'):
            face_tiles[f] = pair

    for spec in args.face:
        # accept forms: "right=1", "+x=3", "top=0"
        if '=' not in spec:
            continue
        k, v = spec.split('=', 1)
        try:
            idx = int(v)
        except ValueError:
            continue
        face_tiles[k.strip()] = idx_to_pair(idx)

    Viewer(args.model, args.elev, args.az, face_tiles).run()


if __name__ == "__main__":
    main()
