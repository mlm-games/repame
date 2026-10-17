//! One place to declare who a game is.
//!
//! Every game used to pass its own triple of strings to the save manager, and
//! six of them passed the same one (`("com", "mlm-games", "WebGame")`), so they
//! shared a data directory and overwrote each other. An [`AppId`] is that
//! triple, spelled once, as a reverse-DNS name.

use std::fmt;
use std::path::PathBuf;

use game_utils::save::SaveManager;
use game_utils::storage::FsStorage;

/// A reverse-DNS application identity, e.g. `com.mlm-games.kickback`.
///
/// Declare it once per game and pass it everywhere a path or a store is built:
///
/// ```rust
/// use repame_save::AppId;
///
/// const KICKBACK: AppId = AppId::new("com.mlm-games.kickback");
///
/// assert_eq!(KICKBACK.app(), "kickback");
/// ```
///
/// The three leading labels are what the platform directory scheme needs:
/// `qualifier`, `organization`, and `application`. Everything after the first
/// dot and before the last is the organization, so `io.github.mlm-games.spin`
/// keeps `github.mlm-games` as one organization.
///
/// A literal id is validated while the crate compiles: a name without three
/// labels, an empty label, or a character outside ASCII letters, digits, `-`
/// and `_` is a compile error, not a save file in the wrong directory.
///
/// Keep the id stable and lowercase. Changing it re-homes every save, setting,
/// and keybind the game has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AppId {
    id: &'static str,
    first_dot: usize,
    last_dot: usize,
}

impl AppId {
    /// An identity from a literal, validated at compile time.
    ///
    /// # Panics
    ///
    /// In a `const` context this is a compile error; at runtime it panics when
    /// `id` is not a reverse-DNS name with at least three labels.
    pub const fn new(id: &'static str) -> Self {
        match split(id) {
            Some((first_dot, last_dot)) => Self {
                id,
                first_dot,
                last_dot,
            },
            None => panic!(
                "repame-save: app id must be reverse-DNS with at least three labels \
                 (`com.example.game`), non-empty labels of ASCII letters, digits, `-` and `_`"
            ),
        }
    }

    /// The whole identity, e.g. `com.mlm-games.kickback`.
    pub fn as_str(&self) -> &'static str {
        self.id
    }

    /// The domain-style qualifier: the first label.
    pub fn qualifier(&self) -> &'static str {
        &self.id[..self.first_dot]
    }

    /// The organization: every label but the first and the last.
    pub fn org(&self) -> &'static str {
        &self.id[self.first_dot + 1..self.last_dot]
    }

    /// The application: the last label.
    pub fn app(&self) -> &'static str {
        &self.id[self.last_dot + 1..]
    }

    /// The platform data directory for this app, created if it is missing.
    ///
    /// The platform rules are `game-utils`' `SaveManager::data_dir`: XDG-style
    /// directories on Linux, `Application Support` on macOS, roaming app data
    /// on Windows, the runtime dir set by the game on Android, and the `ropfs`
    /// (localStorage-backed) path on wasm. Resolved per call, so call it once at
    /// boot rather than in a loop.
    pub fn data_dir(&self) -> PathBuf {
        self.manager("", 0).data_dir()
    }

    /// A file under [`AppId::data_dir`], with the directory created.
    pub fn path(&self, file: &str) -> PathBuf {
        self.data_dir().join(file)
    }

    /// A `game-utils` save manager pointed at this app and file.
    ///
    /// This is the bridge for code that already speaks `SaveManager`: the
    /// identity is still declared in one place, and the file lands in this
    /// app's directory instead of a shared one.
    pub fn manager(&self, file: &'static str, current_version: u32) -> SaveManager<FsStorage> {
        SaveManager::new(
            self.qualifier(),
            self.org(),
            self.app(),
            file,
            current_version,
        )
    }
}

impl fmt::Display for AppId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id)
    }
}

/// Locate the first and last dot of a valid reverse-DNS id.
///
/// Written over bytes so it is callable from a `const` context; the string
/// slices are taken by the accessors, which never allocate.
const fn split(id: &str) -> Option<(usize, usize)> {
    let bytes = id.as_bytes();
    let len = bytes.len();
    let mut first_dot = usize::MAX;
    let mut last_dot = usize::MAX;
    let mut index = 0;
    while index < len {
        let byte = bytes[index];
        if byte == b'.' {
            if first_dot == usize::MAX {
                first_dot = index;
            }
            last_dot = index;
        } else if !(byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_') {
            return None;
        }
        index += 1;
    }
    // Three labels minimum, no leading dot, no trailing dot, no empty middle.
    if first_dot == usize::MAX
        || first_dot == 0
        || first_dot == last_dot
        || first_dot + 1 == last_dot
        || last_dot + 1 == len
    {
        return None;
    }
    Some((first_dot, last_dot))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KICKBACK: AppId = AppId::new("com.mlm-games.kickback");

    #[test]
    fn reverse_dns_decomposes_into_directory_parts() {
        assert_eq!(KICKBACK.as_str(), "com.mlm-games.kickback");
        assert_eq!(KICKBACK.qualifier(), "com");
        assert_eq!(KICKBACK.org(), "mlm-games");
        assert_eq!(KICKBACK.app(), "kickback");
        assert_eq!(KICKBACK.to_string(), "com.mlm-games.kickback");
    }

    #[test]
    fn deeper_ids_keep_the_middle_labels_as_one_organization() {
        let id = AppId::new("io.github.mlm-games.spin");
        assert_eq!(id.qualifier(), "io");
        assert_eq!(id.org(), "github.mlm-games");
        assert_eq!(id.app(), "spin");
    }

    #[test]
    fn manager_carries_the_identity_instead_of_a_shared_app_name() {
        let manager = KICKBACK.manager("kickback.json", 3);
        assert_eq!(manager.qualifier, "com");
        assert_eq!(manager.org, "mlm-games");
        assert_eq!(manager.app, "kickback");
        assert_eq!(manager.file_name, "kickback.json");
        assert_eq!(manager.current_version, 3);
        assert_ne!(manager.app, "WebGame");
    }

    #[test]
    fn distinct_games_have_distinct_identities() {
        let rimshot = AppId::new("com.mlm-games.rimshot");
        assert_ne!(KICKBACK.as_str(), rimshot.as_str());
        assert_ne!(KICKBACK.app(), rimshot.app());
    }
}
