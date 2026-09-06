# repame

Shared crates for writing games on the Repose stack: headless `bevy_ecs` sim,
custom wgpu viewports, Repose UI shell. No Bevy renderer, no `bevy_ui`.

Pattern proven by [resims](../resims): sim snapshot out, pixels in through a
`repose_render_wgpu::Callback` viewport view, everything else is Repose views
(including debug panels via `repose-devtools`).

## Crates

| Crate | Role |
|---|---|
| `repame-sim` | Headless sim: `bevy_ecs` `World` + `Schedule` stepping, fixed-timestep accumulator, snapshot-out convention. Game logic stays engine-shaped and portable to full Bevy. |
| `repame-sprite` | 2D viewport: instanced sprite batch (texture atlas + per-instance transform/uv/color), `Camera2d`, CPU picking, fullscreen postfx hook. Covers the 2D games (rozvp, Opensus, nt-recreated-bevy, Floppy-Warriors) and 2D remakes. |
| `repame-shell` | App wiring: Repose platform runners + sim stepping + viewport mount, gamepad (`gilrs`) to input mapping, save-path helpers. |

Planned, not yet scaffolded: `repame-view3d` (glTF/skinning/PBR viewport for rustbox + 3D remakes).

## Pilot

**rozvp** (PvZ clone): smallest render surface (20 sprites, no physics/gamepad),
migrates behind a `repose-shell` feature strangler-style. UI is already Repose
via `repose-bevy`, so the pilot is nearly pure viewport swap.



## License

TODO: pick license before first push (resims is GPL-3.0, repose is MPL-2.0).
