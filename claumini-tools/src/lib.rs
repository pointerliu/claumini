#![forbid(unsafe_code)]

mod adapters;
mod metadata;
mod repo;

pub use adapters::AsyncToolAdapter;
pub use adapters::SyncToolAdapter;
pub use claumini_tools_macros::ToolRegistration;
pub use metadata::{ToolMetadata, ToolRegistration};
pub use repo::{
    ListFilesInput, ListFilesOutput, ListFilesTool, ReadFileInput, ReadFileOutput, ReadFileTool,
    ReadRangeInput, ReadRangeLine, ReadRangeOutput, ReadRangeTool, SearchMatch, SearchTextInput,
    SearchTextOutput, SearchTextTool,
};

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use claumini_core::{SessionMetadata, Tool, ToolContext, ToolError};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use super::{
        ListFilesInput, ListFilesTool, ReadFileInput, ReadFileTool, ReadRangeInput, ReadRangeTool,
        SearchTextInput, SearchTextTool, SyncToolAdapter, ToolMetadata,
    };

    fn tool_context() -> ToolContext {
        ToolContext::new(SessionMetadata::root("test-session"), 0)
    }

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent directories should be created");
        }
        fs::write(path, contents).expect("test file should be written");
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
    struct SchemaInput {
        pub query: String,
        pub limit: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
    struct SchemaOutput {
        pub matches: Vec<String>,
    }

    #[test]
    fn typed_metadata_descriptor_populates_input_and_output_schemas() {
        let descriptor = ToolMetadata::new("search", "Searches text")
            .descriptor_for::<SchemaInput, SchemaOutput>();

        assert_eq!(descriptor.input_schema["type"], "object");
        assert!(descriptor.input_schema["properties"].get("query").is_some());
        assert!(descriptor.input_schema["properties"].get("limit").is_some());
        assert_eq!(
            descriptor.output_schema.as_ref().expect("output schema")["type"],
            "object"
        );
        assert!(
            descriptor.output_schema.as_ref().expect("output schema")["properties"]
                .get("matches")
                .is_some()
        );
    }

    #[tokio::test]
    async fn sync_tool_adapter_exposes_metadata_and_runs_sync_logic() {
        let tool = SyncToolAdapter::new(
            ToolMetadata::new("uppercase", "Uppercases the input"),
            |input: String| Ok(input.to_uppercase()),
        );

        let descriptor = tool.descriptor();
        let output = tool
            .call(String::from("hello"), &mut tool_context())
            .await
            .expect("sync adapter should execute");

        assert_eq!(descriptor.name, "uppercase");
        assert_eq!(descriptor.description, "Uppercases the input");
        assert_eq!(descriptor.input_schema["type"], "string");
        assert_eq!(
            descriptor.output_schema.as_ref().expect("output schema")["type"],
            "string"
        );
        assert_eq!(output, "HELLO");
    }

    #[test]
    fn repo_tool_descriptors_expose_generated_schemas() {
        let descriptor = SearchTextTool::new().descriptor();

        assert_eq!(descriptor.input_schema["type"], "object");
        assert!(descriptor.input_schema["properties"].get("root").is_some());
        assert_eq!(
            descriptor.input_schema["properties"]["root"]["description"],
            "Root directory whose UTF-8 files will be searched."
        );
        assert!(
            descriptor.input_schema["properties"]
                .get("pattern")
                .is_some()
        );
        assert_eq!(
            descriptor.input_schema["properties"]["pattern"]["description"],
            "Regular expression pattern to search for in each file."
        );
        assert_eq!(
            descriptor.output_schema.as_ref().expect("output schema")["type"],
            "object"
        );
        assert!(
            descriptor.output_schema.as_ref().expect("output schema")["properties"]
                .get("matches")
                .is_some()
        );
        assert_eq!(
            descriptor.output_schema.as_ref().expect("output schema")["properties"]["matches"]["description"],
            "Matching lines found under the root directory."
        );
    }

    #[tokio::test]
    async fn list_files_returns_sorted_relative_paths() {
        let temp = tempdir().expect("temp dir should exist");
        write_file(&temp.path().join("src/lib.rs"), "pub fn lib() {}\n");
        write_file(&temp.path().join("README.md"), "# demo\n");

        let output = ListFilesTool::new()
            .call(
                ListFilesInput {
                    root: temp.path().to_path_buf(),
                },
                &mut tool_context(),
            )
            .await
            .expect("list_files should succeed");

        assert_eq!(output.paths, vec!["README.md", "src/lib.rs"]);
    }

    #[tokio::test]
    async fn search_text_returns_matching_lines_with_line_numbers() {
        let temp = tempdir().expect("temp dir should exist");
        let file_path = temp.path().join("src/main.rs");
        write_file(&file_path, "fn main() {}\nlet needle = 1;\nneedle += 1;\n");

        let output = SearchTextTool::new()
            .call(
                SearchTextInput {
                    root: temp.path().to_path_buf(),
                    pattern: String::from("needle"),
                },
                &mut tool_context(),
            )
            .await
            .expect("search_text should succeed");

        assert_eq!(output.matches.len(), 2);
        assert_eq!(output.matches[0].path, "src/main.rs");
        assert_eq!(output.matches[0].line_number, 2);
        assert_eq!(output.matches[1].line_number, 3);
    }

    #[tokio::test]
    async fn search_text_skips_non_utf8_files_and_keeps_text_matches() {
        let temp = tempdir().expect("temp dir should exist");
        write_file(
            &temp.path().join("src/lib.rs"),
            "const NEEDLE: &str = \"needle\";\n",
        );
        let binary_path = temp.path().join("assets/blob.bin");
        fs::create_dir_all(
            binary_path
                .parent()
                .expect("binary file should have a parent"),
        )
        .expect("binary file parent should be created");
        fs::write(binary_path, [0xff, 0xfe, 0xfd]).expect("binary test file should be written");

        let output = SearchTextTool::new()
            .call(
                SearchTextInput {
                    root: temp.path().to_path_buf(),
                    pattern: String::from("needle"),
                },
                &mut tool_context(),
            )
            .await
            .expect("search_text should skip unreadable files");

        assert_eq!(output.matches.len(), 1);
        assert_eq!(output.matches[0].path, "src/lib.rs");
        assert_eq!(output.matches[0].line_number, 1);
        assert_eq!(output.matches[0].line, "const NEEDLE: &str = \"needle\";");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn list_files_reports_traversal_errors_instead_of_succeeding() {
        let temp = tempdir().expect("temp dir should exist");
        let blocked_dir = temp.path().join("blocked");
        fs::create_dir(&blocked_dir).expect("blocked dir should exist");

        let original_permissions = fs::metadata(&blocked_dir)
            .expect("blocked dir metadata should exist")
            .permissions();
        let mut blocked_permissions = original_permissions.clone();
        blocked_permissions.set_mode(0o000);
        fs::set_permissions(&blocked_dir, blocked_permissions)
            .expect("blocked dir permissions should update");

        let error = ListFilesTool::new()
            .call(
                ListFilesInput {
                    root: temp.path().to_path_buf(),
                },
                &mut tool_context(),
            )
            .await
            .expect_err("list_files should fail on traversal errors");

        fs::set_permissions(&blocked_dir, original_permissions)
            .expect("blocked dir permissions should restore");

        assert!(
            matches!(error, ToolError::ExecutionFailed(message) if message.contains("failed to walk directory"))
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn search_text_reports_non_utf8_unrelated_read_errors() {
        let temp = tempdir().expect("temp dir should exist");
        let unreadable_path = temp.path().join("secret.txt");
        write_file(&unreadable_path, "needle\n");

        let original_permissions = fs::metadata(&unreadable_path)
            .expect("secret file metadata should exist")
            .permissions();
        let mut unreadable_permissions = original_permissions.clone();
        unreadable_permissions.set_mode(0o000);
        fs::set_permissions(&unreadable_path, unreadable_permissions)
            .expect("secret file permissions should update");

        let error = SearchTextTool::new()
            .call(
                SearchTextInput {
                    root: temp.path().to_path_buf(),
                    pattern: String::from("needle"),
                },
                &mut tool_context(),
            )
            .await
            .expect_err("search_text should fail on unreadable utf-8 files");

        fs::set_permissions(&unreadable_path, original_permissions)
            .expect("secret file permissions should restore");

        assert!(
            matches!(error, ToolError::ExecutionFailed(message) if message.contains("failed to read file"))
        );
    }

    #[tokio::test]
    async fn read_file_returns_full_contents() {
        let temp = tempdir().expect("temp dir should exist");
        let file_path = temp.path().join("notes.txt");
        write_file(&file_path, "alpha\nbeta\n");

        let output = ReadFileTool::new()
            .call(ReadFileInput { path: file_path }, &mut tool_context())
            .await
            .expect("read_file should succeed");

        assert_eq!(output.contents, "alpha\nbeta\n");
    }

    #[tokio::test]
    async fn read_range_returns_requested_lines_only() {
        let temp = tempdir().expect("temp dir should exist");
        let file_path = temp.path().join("notes.txt");
        write_file(&file_path, "line 1\nline 2\nline 3\nline 4\n");

        let output = ReadRangeTool::new()
            .call(
                ReadRangeInput {
                    path: file_path,
                    start_line: 2,
                    end_line: 3,
                },
                &mut tool_context(),
            )
            .await
            .expect("read_range should succeed");

        assert_eq!(output.lines.len(), 2);
        assert_eq!(output.lines[0].number, 2);
        assert_eq!(output.lines[0].text, "line 2");
        assert_eq!(output.lines[1].number, 3);
        assert_eq!(output.lines[1].text, "line 3");
    }

    #[tokio::test]
    async fn read_range_rejects_invalid_ranges() {
        let temp = tempdir().expect("temp dir should exist");
        let file_path = temp.path().join("notes.txt");
        write_file(&file_path, "line 1\nline 2\n");

        let error = ReadRangeTool::new()
            .call(
                ReadRangeInput {
                    path: file_path,
                    start_line: 3,
                    end_line: 2,
                },
                &mut tool_context(),
            )
            .await
            .expect_err("invalid ranges should fail");

        assert!(
            matches!(error, ToolError::InvalidInput(message) if message.contains("start_line"))
        );
    }
}
