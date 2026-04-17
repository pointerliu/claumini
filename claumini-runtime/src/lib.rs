#![forbid(unsafe_code)]

mod artifacts;
mod prompt_agent;
mod runtime_tools;
mod session;
mod skills;

pub use artifacts::{ArtifactBody, ArtifactRecord, ArtifactStore};
pub use prompt_agent::{
    ChildContextMode, ChildRegistration, ChildToolPolicy, PromptAgent, PromptAgentBuilder,
    PromptSession, ReservedRuntimeTools, RuntimeTool,
};
pub use runtime_tools::{
    AWAIT_SUBAGENT_TOOL_NAME, CALL_SUBAGENT_TOOL_NAME, FINISH_TOOL_NAME, HANDOFF_TOOL_NAME,
    LOAD_SKILL_TOOL_NAME, RESERVED_TOOL_NAMES, SPAWN_SUBAGENT_TOOL_NAME, reserved_tool_names,
};
pub use session::{ChildHandleRecord, SessionRecord, SessionState, ToolCallRecord};
pub use skills::{SkillMetadata, SkillRegistry};

/// Build a `String` user message from a struct by binding named fields into a
/// `format!` template.
///
/// Equivalent to `format!($tmpl, field = input.field, ...)`. Placeholder names
/// in the template must match the listed field names (checked by the compiler).
#[macro_export]
macro_rules! prompt_template {
    ($tmpl:literal, $input:expr, { $($field:ident),+ $(,)? }) => {{
        let __input = &$input;
        format!($tmpl, $($field = &__input.$field),+)
    }};
}

#[cfg(test)]
mod tests {
    use std::fs;

    use claumini_core::Payload;
    use tempfile::tempdir;

    use super::{
        ArtifactBody, ArtifactStore, SessionRecord, SessionState, SkillRegistry,
        reserved_tool_names,
    };

    #[test]
    fn artifact_store_round_trips_records_and_allocates_unique_ids() {
        let store = ArtifactStore::new();

        let payload_id = store
            .insert_payload(Payload::text("report body"))
            .expect("payload artifact should store");
        let bytes_id = store.insert_bytes(Some("report.txt"), Some("text/plain"), b"hello world");

        assert_ne!(payload_id, bytes_id);

        let payload = store
            .get(payload_id)
            .expect("payload artifact should exist");
        assert_eq!(payload.id, payload_id);
        assert_eq!(
            payload.body,
            ArtifactBody::Payload(Payload::text("report body"))
        );

        let bytes = store.get(bytes_id).expect("byte artifact should exist");
        assert_eq!(bytes.name.as_deref(), Some("report.txt"));
        assert_eq!(bytes.media_type.as_deref(), Some("text/plain"));
        assert_eq!(bytes.body, ArtifactBody::Bytes(b"hello world".to_vec()));
    }

    #[test]
    fn session_record_starts_running_with_empty_runtime_state() {
        let session = SessionRecord::new("session-1");

        assert_eq!(session.state, SessionState::Running);
        assert_eq!(session.metadata.session_id, "session-1");
        assert_eq!(session.current_input, None);
        assert!(session.transcript.is_empty());
        assert!(session.tool_calls.is_empty());
        assert!(session.artifact_ids.is_empty());
        assert!(session.children.is_empty());
    }

    #[test]
    fn session_record_owns_current_input() {
        let mut session = SessionRecord::new("session-1");
        session.current_input = Some(Payload::text("current input"));

        assert_eq!(session.current_input, Some(Payload::text("current input")));
    }

    #[test]
    fn prompt_template_substitutes_struct_fields() {
        struct Case {
            a: String,
            b: u32,
        }
        let c = Case {
            a: "hi".to_string(),
            b: 7,
        };
        let s = crate::prompt_template!("{a}/{b}", c, { a, b });
        assert_eq!(s, "hi/7");
        // struct still usable after macro — fields were borrowed, not moved.
        assert_eq!(c.a, "hi");
        assert_eq!(c.b, 7);
    }

    #[test]
    fn reserved_runtime_tools_are_stable() {
        assert_eq!(
            reserved_tool_names(),
            [
                "call_subagent",
                "spawn_subagent",
                "await_subagent",
                "handoff",
                "load_skill",
                "finish",
            ]
        );
    }

    #[test]
    fn skill_registry_scans_skill_markdown_and_loads_bodies_on_demand() {
        let temp = tempdir().expect("tempdir should exist");
        let alpha_dir = temp.path().join("alpha-skill");
        let beta_dir = temp.path().join("nested").join("beta-skill");

        fs::create_dir_all(&alpha_dir).expect("alpha skill dir should exist");
        fs::create_dir_all(&beta_dir).expect("beta skill dir should exist");

        fs::write(
            alpha_dir.join("SKILL.md"),
            "# Alpha Skill\n\nAlpha description.\n\nFull alpha body.\n",
        )
        .expect("alpha skill file should be written");
        fs::write(
            beta_dir.join("SKILL.md"),
            "# Beta Skill\n\nBeta description.\n\nFull beta body.\n",
        )
        .expect("beta skill file should be written");

        let registry = SkillRegistry::scan([temp.path()]).expect("skill scan should succeed");
        let skills = registry.metadata();

        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "alpha-skill");
        assert_eq!(skills[0].description, "Alpha description.");
        assert_eq!(skills[1].name, "beta-skill");
        assert_eq!(skills[1].description, "Beta description.");

        let beta_body = registry
            .load("beta-skill")
            .expect("beta skill body should load");
        assert!(beta_body.contains("Full beta body."));
    }

    #[cfg(unix)]
    #[test]
    fn skill_registry_scan_propagates_walkdir_traversal_errors() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempdir().expect("tempdir should exist");
        let blocked_dir = temp.path().join("blocked");
        fs::create_dir(&blocked_dir).expect("blocked dir should exist");

        let original_permissions = fs::metadata(&blocked_dir)
            .expect("blocked dir metadata should exist")
            .permissions();
        let mut blocked_permissions = original_permissions.clone();
        blocked_permissions.set_mode(0o000);
        fs::set_permissions(&blocked_dir, blocked_permissions)
            .expect("blocked dir permissions should update");

        let error = SkillRegistry::scan([temp.path()]).expect_err("scan should fail");

        fs::set_permissions(&blocked_dir, original_permissions)
            .expect("blocked dir permissions should restore");

        assert!(
            error.to_string().contains("failed to walk skills root"),
            "unexpected error: {error}"
        );
    }
}
