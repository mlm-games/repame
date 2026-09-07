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
| `repame-sprite` | 2D viewport: instanced sprite batch (texture atlas + per-instance transform/uv/color), `Camera2d`, CPU picking, fullscreen postfx hook. Atm mainly aiming to cover my 2D games (rozvp, Opensus, Floppy-Warriors's rust version) and 2D remakes (nt-recreated-bevy) and stabilise later once core bugs are fixed. |
| `repame-shell` | App wiring: Repose platform runners + sim stepping + viewport mount, gamepad to input mapping (done internally in repose), save-path helpers, etc. |

Planned, not yet scaffolded: `repame-view3d` (glTF/skinning/PBR viewport for rustbox, and for more 3D games later). Should be usable externally too, but do consider, that it is in a very early phase, and might have many bugs (though the api might not change heavily)

# Internal notes

## Pilot

rozvp (PvZ clone): UI was already Repose via `repose-bevy`, so the pilot is nearly pure viewport swap. Old branch is preserved for reference to a `repose-bevy` game.

## License

MPL-2.0
