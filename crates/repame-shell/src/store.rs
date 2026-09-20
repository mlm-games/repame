use std::path::{Path, PathBuf};

pub fn save_file_path(env_var: &str, file_name: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_var)
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(file_name)
}

pub fn store_json(text: &str, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // Crash-safe: temp + rename, so a mid-write kill leaves either the old
    // file or the new one, never a torn half-write. Same-volume `rename`
    // replaces atomically on POSIX and on Windows (std sets
    // `MOVEFILE_REPLACE_EXISTING`); the temp lives next to the target so no
    // cross-volume copy. No `fsync`: an OS crash (not just a process kill)
    // can still lose the rename; games needing power-loss durability must
    // fsync the temp and the directory game-side.
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

pub fn load_json(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_wins() {
        let _lock = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("REPAME_SAVE_TEST_PATH", "/tmp/x.json") };
        assert_eq!(
            save_file_path("REPAME_SAVE_TEST_PATH", "s.json"),
            PathBuf::from("/tmp/x.json")
        );
        unsafe { std::env::remove_var("REPAME_SAVE_TEST_PATH") };
    }

    #[test]
    fn round_trip_leaves_no_tmp() {
        let dir = std::env::temp_dir().join(format!("repame-store-{}", std::process::id()));
        let path = dir.join("save.json");
        let _ = std::fs::remove_dir_all(&dir);
        store_json("{\"a\":1}", &path).expect("write");
        assert_eq!(load_json(&path).expect("read"), "{\"a\":1}");
        assert!(
            !path.with_extension("tmp").exists(),
            "temp renamed away, not left behind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
