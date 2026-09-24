use std::env;
use std::process::ExitCode;

use bevy_ecs::prelude::World;
use repame_content::Project;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let command = args.next();
    let root = args.next();
    if args.next().is_some() {
        return usage();
    }

    match (command.as_deref(), root) {
        (Some("validate"), Some(root)) => match Project::load(root) {
            Ok(project) => {
                println!(
                    "{}: {} scenes, {} assets, {} resources",
                    project.manifest().name,
                    project.scene_count(),
                    project.asset_count(),
                    project.resource_count()
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        },
        (Some("load"), Some(root)) => match Project::load(root) {
            Ok(project) => {
                let mut world = World::new();
                match project.spawn_entry_scene(&mut world) {
                    Ok(scene) => {
                        println!(
                            "{}: loaded `{}` with {} entities",
                            project.manifest().name,
                            scene.scene_id,
                            scene.entities.len()
                        );
                        ExitCode::SUCCESS
                    }
                    Err(error) => {
                        eprintln!("{error}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        },
        _ => usage(),
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: repame-content <validate|load> <project-directory>");
    ExitCode::from(2)
}
