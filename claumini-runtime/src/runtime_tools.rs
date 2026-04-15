pub const CALL_SUBAGENT_TOOL_NAME: &str = "call_subagent";
pub const SPAWN_SUBAGENT_TOOL_NAME: &str = "spawn_subagent";
pub const AWAIT_SUBAGENT_TOOL_NAME: &str = "await_subagent";
pub const HANDOFF_TOOL_NAME: &str = "handoff";
pub const LOAD_SKILL_TOOL_NAME: &str = "load_skill";
pub const FINISH_TOOL_NAME: &str = "finish";

pub const RESERVED_TOOL_NAMES: [&str; 6] = [
    CALL_SUBAGENT_TOOL_NAME,
    SPAWN_SUBAGENT_TOOL_NAME,
    AWAIT_SUBAGENT_TOOL_NAME,
    HANDOFF_TOOL_NAME,
    LOAD_SKILL_TOOL_NAME,
    FINISH_TOOL_NAME,
];

pub const fn reserved_tool_names() -> [&'static str; 6] {
    RESERVED_TOOL_NAMES
}
