# repame

Shared crates for writing games on the Repose stack: headless `bevy_ecs` sim,
custom wgpu viewports, and a repose UI shell. Bevy renderer is not used here, mainly for:
1. wgpu internal version rivalry.
2. more dogfooding.

Pattern initially tested using [resims](https://github.com/mlm-games/resims)'s old branch, which used sim snapshot out, pixels in through a `repose_render_wgpu::Callback` viewport view, everything else should be Repose views (including debug panels).

## Crates

| Crate | Role |
|---|---|
| `repame-sim` | Headless sim: `bevy_ecs` `World` + `Schedule` stepping, fixed-timestep accumulator, snapshot-out convention. |
| `repame-atlas` | CPU shelf atlas packing + upload queue (`UvRect`, drain-once uploads). |
| `repame-anim` | Frame-strip animation player (loop/once/ping-pong). |
| `repame-audio` | Audio banks / rigs / music wiring. |
| `repame-sprite` | 2D viewport: instanced sprite batch (texture atlas + per-instance transform/uv/color), `Camera2d`, CPU picking, fullscreen postfx hook. Atm mainly aiming to cover my 2D games (rozvp, Opensus, Floppy-Warriors's rust version) and 2D remakes (nt-recreated-bevy) and stabilise later once core bugs are fixed. |
| `repame-shell` | App wiring: Repose platform runners + sim stepping + viewport mount, gamepad to input mapping (done internally in repose), save-path helpers, etc. |
| `repame-actors` | Live vector actors (`.ren` rigs via `renamite-player`). |
| `repame-fx` | Sim-side particles (2D sprites + 3D billboards), ground decals + blob shadows, trauma shake, flash, floaters, transitions. |
| `repame-input` | Leafwing-shaped action map / `ActionState` edges-at-tick-end. |
| `repame-view3d` | 3D viewport: orbit camera, mesh snapshots (flat/lit/textured, PBR-lite material + fog + tonemap), depth-tested wgpu pass, CPU mesh picking, glTF static import (materials/alpha) + CPU skinning/animation tracks/morphs + bone attachments, voxel chunk mesher (greedy Full faces, rotation-aware occlusion, parity-proven vs `rustbox-mesh`), dirty-tracked chunk cache. Long-term foundation (shadows later). |
| `repame-physics` | Sim-side character/voxel physics: AABB mover over `VoxelQuery` + pushback, `repame-sim` systems + transform snapshots. |

Planned, not yet scaffolded: GPU shadow maps (blob shadows via `repame-fx` decals cover the grounding read; maps plug in behind `repame-view3d`'s `MeshGroup` / `Frame3d` if a game ever needs them).

# Internal notes

## Pilot

rozvp (PvZ clone): UI was already Repose via `repose-bevy`, so the pilot is nearly pure viewport swap. Old branch is preserved for reference to a `repose-bevy` game.

## License

MPL-2.0
