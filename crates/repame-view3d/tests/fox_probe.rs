/// Fixture roots, first hit wins. Absolute local paths are last-resort
/// fallbacks for this author's machine; CI and fresh clones skip instead
/// of failing (same pattern as the gnome test in `skin.rs`).
fn fixture_bytes(name: &str) -> Option<Vec<u8>> {
    let candidates = [
        format!("crates/repame-view3d/tests/assets/{name}"),
        format!("/home/ymsr/Documents/Repos/Old-Fyrox/bevy/assets/models/animated/{name}"),
    ];
    for path in &candidates {
        match std::fs::read(path) {
            Ok(bytes) => return Some(bytes),
            Err(e) => eprintln!("SKIP fixture {name} ({path}): {e}"),
        }
    }
    None
}

#[test]
fn fox_probe() {
    let Some(bytes) = fixture_bytes("Fox.glb") else {
        eprintln!("SKIP fox_probe (no fixture)");
        return;
    };
    let skinned = repame_view3d::import_skinned(&bytes).expect("fox parses");
    assert_eq!(skinned.len(), 1, "one skinned mesh");
    let m = &skinned[0];
    assert_eq!(m.positions.len(), 1728);
    assert_eq!(m.joint_count(), 24);
    assert_eq!(m.node_to_joint.len(), 24);
    let skel = repame_view3d::import_skeleton(&bytes).expect("skel");
    assert_eq!(skel.node_count(), 26, "scene graph nodes");
    let anims = repame_view3d::import_animations(&bytes).expect("anims");
    assert_eq!(anims.len(), 3);
    assert_eq!(anims[0].name, "Survey");
    assert_eq!(anims[1].name, "Walk");
    assert_eq!(anims[2].name, "Run");
    // Hierarchy: Walk moves every vert through the composed tree.
    let a = &anims[1];
    let bind = m.pose(&vec![glam::Mat4::IDENTITY; m.joint_count()]);
    assert_eq!(bind.tri_count(), 576, "fox is single-sided");
    let posed = m.pose(&a.joint_matrices(&skel, m, a.duration() / 2.0));
    let moved = posed
        .positions
        .iter()
        .zip(bind.positions.iter())
        .filter(|(x, y)| {
            (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2) > 1e-10
        })
        .count();
    assert_eq!(moved, posed.positions.len(), "Walk moves the whole fox");
    // Player agrees with the manual path.
    let mut p = repame_view3d::SkeletonPlayer::new(a, repame_view3d::SkeletonLoop::Loop);
    p.play();
    p.seek(a.duration() / 2.0);
    let via = p.joint_matrices(a, &skel, m);
    let manual = a.joint_matrices(&skel, m, a.duration() / 2.0);
    for (x, y) in via.iter().zip(manual.iter()) {
        assert!(
            x.to_cols_array()
                .iter()
                .zip(y.to_cols_array().iter())
                .all(|(u, v)| (u - v).abs() < 1e-6)
        );
    }
}

#[test]
fn morph_stress_imports_and_blends() {
    let Some(bytes) = fixture_bytes("MorphStressTest.gltf") else {
        eprintln!("SKIP morph_stress_imports_and_blends (no fixture)");
        return;
    };
    let sets = repame_view3d::import_morphs(&bytes).expect("morphs parse");
    assert_eq!(sets.len(), 1);
    assert_eq!(sets[0].target_count(), 8, "eight targets");
    let n: usize = sets[0].position_deltas.iter().map(|d| d.len()).sum();
    assert!(n > 0, "has deltas");
    // NOTE: this fixture's deltas are all zeros (min == max == 0: the
    // stress test animates weights only). Import + track assertions below
    // still pin the path; `morph_set_blends_positions_and_renormalizes`
    // proves real deltas blend.
    let anims = repame_view3d::import_animations(&bytes).expect("morph anims parse");
    assert_eq!(anims.len(), 3, "Individuals/Pulse/TheWave");
    let pulse = anims.iter().find(|a| a.name == "Pulse").expect("Pulse");
    assert_eq!(pulse.morph_tracks.len(), 1, "one mesh track");
    let (mesh, track) = pulse.morph_tracks.iter().next().unwrap();
    assert_eq!(*mesh, 0);
    assert_eq!(track.target_count, 8);
    assert_eq!(track.times.len(), 153, "153 keys");
    // Ends hold the extremes (clamp), the middle moves between them.
    assert_eq!(track.sample(0.0), track.sample(-1.0), "start clamps");
    let mid = track.sample(track.duration() / 2.0);
    assert_eq!(mid.len(), 8);
    assert!(mid.iter().all(|w| (0.0..=1.0).contains(w)), "{mid:?}");
    // Apply the end weights (all zeros here. Pulse returns to rest):
    // with nonzero weights the blend must move verts. This fixture's
    // deltas are all zeros (min == max == 0), so assert the no-op
    // direction: zero deltas never move verts even at full weight.
    let n = sets[0].position_deltas[0].len();
    let mut group = repame_view3d::MeshGroup {
        depth_test: true,
        ..Default::default()
    };
    for i in 0..n {
        group.positions.push([i as f32, 0.0, 0.0]);
        group.colors.push([1.0, 1.0, 1.0]);
    }
    group.indices = (0..n as u32).collect();
    let before = group.positions.clone();
    sets[0].apply(&mut group, &[1.0; 8]);
    assert_eq!(
        group.positions, before,
        "zero-delta fixture stays put at full weight"
    );
}
