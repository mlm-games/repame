# repame

Shared crates for writing games on the Repose stack: headless `bevy_ecs` sim,
custom wgpu viewports, and a repose UI shell. Bevy renderer is not used here, mainly for:
1. wgpu internal version rivalry.
2. more dogfooding.

Pattern initially tested using [resims](https://github.com/mlm-games/resims)'s old branch, which used sim snapshot out, pixels in through a `repose_render_wgpu::Callback` viewport view, everything else was Repose views (including debug panels).

## Crates

| Crate | Role |
|---|---|
| `repame-sim` | Headless sim: `bevy_ecs` world, fixed-step accumulator, deterministic RNG, world hasher, session tape. |
| `repame-content` | Content foundation: project manifest, import pipeline, asset VFS, DDS/KTX codecs, package archives. |
| `repame-atlas` | CPU shelf atlas packing + upload queue (`UvRect`, drain-once uploads). |
| `repame-anim` | Frame-strip animation player (loop/once/ping-pong). |
| `repame-audio` | Sound banks, buses, effects, spatial audio, and declarative cue synthesis. |
| `repame-view2d` | 2D viewport: instanced sprite batch, camera fit, picking, raster sheets, world text, vector actors, lighting. |
| `repame-view3d` | 3D viewport: orbit camera, mesh snapshots, glTF import, chunk streaming, shadow maps, texture streaming. |
| `repame-shell` | App wiring: platform runners, gamepad, host helpers, and the portal integration. |
| `repame-fx` | Sim-side particles, decals, blob shadows, trauma shake, flash, floaters, transitions. |
| `repame-input` | Leafwing-shaped action map / `ActionState` edges-at-tick-end. |
| `repame-physics` | Sim-side character/voxel physics: AABB mover over `VoxelQuery` + pushback, transform snapshots. |
| `repame-save` | One app identity, one versioned save envelope, one place for settings, keybinds and sim state. |
| `repame-ui` | Shared UI kit: action queue, run flow, focus navigation, palette, touch overlay. |
| `repame-reflect` | ECS reflection: component registry, RON world snapshot, inspectable component tree. |
| `repame-nav` | Spatial navigation: baked navmesh, A* queries, funnel corridors, path-following agents. |
| `repame-net` | Networking: transports, channels, replication, RPC. |
| `repame-l10n` | Runtime localization: message catalogs from RON or Godot-style `.po`. |
| `repame-ik` | Inverse kinematics and skeleton modifiers over view3d skeletons. |
| `repame-xr` | XR: session lifecycle, reference spaces, stereo views, action sets. |
| `repame-rapier2d` | Rapier physics over the sim: 2D bodies, joints, rays, snapshots. |
| `repame-rapier3d` | Rapier physics over the sim: 3D bodies, joints, rays, snapshots. |

# Internal notes

## 2D coverage

The 2D viewport mainly aims to cover the 2D games (rozvp, Opensus, the rust
version of Floppy-Warriors) and 2D remakes (nt-recreated-bevy), and stabilises
later once the core bugs are fixed.

## Pilot

rozvp (PvZ clone): UI was already Repose via `repose-bevy`, so the pilot is nearly pure viewport swap. Old branch is preserved for reference to a `repose-bevy` game.

## License

MPL-2.0
