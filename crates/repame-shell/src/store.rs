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
    std::fs::write(path, text).map_err(|e| e.to_string())
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

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
