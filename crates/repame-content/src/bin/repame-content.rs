use std::env;
use std::process::ExitCode;

use bevy_ecs::prelude::World;
use repame_content::{AssetPipeline, ContentError, Project};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("validate") => project_command(&args, |project| {
            println!(
                "{}: {} scenes, {} assets, {} resources",
                project.manifest().name,
                project.scene_count(),
                project.asset_count(),
                project.resource_count()
            );
            Ok(())
        }),
        Some("import") => project_command(&args, |project| {
            let imported = project.import_assets()?;
            let reused = imported.iter().filter(|asset| asset.reused).count();
            println!(
                "{}: imported {} assets ({} reused)",
                project.manifest().name,
                imported.len(),
                reused
            );
            Ok(())
        }),
        Some("load") => project_command(&args, |project| {
            let mut world = World::new();
            let scene = project.spawn_entry_scene(&mut world)?;
            println!(
                "{}: loaded `{}` with {} entities",
                project.manifest().name,
                scene.scene_id,
                scene.entities.len()
            );
            Ok(())
        }),
        Some("pack") => pack_command(&args),
        _ => usage(),
    }
}

fn project_command(
    args: &[String],
    command: impl FnOnce(&Project) -> Result<(), ContentError>,
) -> ExitCode {
    let (Some(root), None) = (args.get(1), args.get(2)) else {
        return usage();
    };
    match Project::load(root).and_then(|project| command(&project).map(|()| project)) {
        Ok(_) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

/// `pack <source-directory> [output-directory]`: the same entry point a game
/// calls, so a Python or shell content tool can run the pipeline without
/// linking Rust.
fn pack_command(args: &[String]) -> ExitCode {
    let (Some(source_root), output_root, None) = (args.get(1), args.get(2), args.get(3)) else {
        return usage();
    };
    let output_root = output_root.map(String::as_str).unwrap_or(source_root);
    let build = AssetPipeline::new(source_root).and_then(|pipeline| pipeline.build(output_root));
    match build {
        Ok(build) => {
            println!(
                "{} entries, {} bytes of sources, {} bytes packed into {} (+ {})",
                build.entries(),
                build.manifest.total_size(),
                build.archive_size,
                build.archive_path.display(),
                build.manifest_path.display()
            );
            if let Some(atlas) = build.manifest.atlas() {
                println!(
                    "atlas: {} pages of {} px, {} sprites",
                    atlas.pages,
                    atlas.page_size,
                    atlas.len()
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: repame-content <validate|import|load> <project-directory>\n       repame-content pack <source-directory> [output-directory]"
    );
    ExitCode::from(2)
}
