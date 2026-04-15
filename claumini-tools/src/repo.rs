use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use claumini_core::{Tool, ToolContext, ToolDescriptor, ToolError};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::ToolMetadata;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ListFilesInput {
    #[schemars(description = "Root directory to recursively list files from.")]
    pub root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ListFilesOutput {
    #[schemars(description = "Sorted relative file paths found under the root directory.")]
    pub paths: Vec<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ListFilesTool;

impl ListFilesTool {
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ListFilesTool {
    type Input = ListFilesInput;
    type Output = ListFilesOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolMetadata::new("list_files", "Lists files under a root directory")
            .descriptor_for::<ListFilesInput, ListFilesOutput>()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        let root = require_directory(&input.root)?;
        let mut paths = Vec::new();
        for entry in WalkDir::new(root) {
            let entry = entry.map_err(walkdir_error)?;
            if entry.file_type().is_file() {
                paths.push(relative_path(root, entry.path())?);
            }
        }
        paths.sort();

        Ok(ListFilesOutput { paths })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchTextInput {
    #[schemars(description = "Root directory whose UTF-8 files will be searched.")]
    pub root: PathBuf,
    #[schemars(description = "Regular expression pattern to search for in each file.")]
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchMatch {
    #[schemars(description = "Relative path of the file containing the match.")]
    pub path: String,
    #[schemars(description = "1-based line number containing the match.")]
    pub line_number: usize,
    #[schemars(description = "Full text of the matching line.")]
    pub line: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchTextOutput {
    #[schemars(description = "Matching lines found under the root directory.")]
    pub matches: Vec<SearchMatch>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchTextTool;

impl SearchTextTool {
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for SearchTextTool {
    type Input = SearchTextInput;
    type Output = SearchTextOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolMetadata::new(
            "search_text",
            "Searches file contents under a root directory",
        )
        .descriptor_for::<SearchTextInput, SearchTextOutput>()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        let root = require_directory(&input.root)?;
        let pattern = Regex::new(&input.pattern)
            .map_err(|error| ToolError::InvalidInput(format!("invalid regex pattern: {error}")))?;

        let mut matches = Vec::new();
        for entry in WalkDir::new(root) {
            let entry = entry.map_err(walkdir_error)?;
            if !entry.file_type().is_file() {
                continue;
            }

            let contents = match fs::read_to_string(entry.path()) {
                Ok(contents) => contents,
                Err(error) if error.kind() == io::ErrorKind::InvalidData => continue,
                Err(error) => {
                    return Err(ToolError::ExecutionFailed(format!(
                        "failed to read file '{}': {error}",
                        entry.path().display()
                    )));
                }
            };
            let relative = relative_path(root, entry.path())?;
            for (line_index, line) in contents.lines().enumerate() {
                if pattern.is_match(line) {
                    matches.push(SearchMatch {
                        path: relative.clone(),
                        line_number: line_index + 1,
                        line: line.to_string(),
                    });
                }
            }
        }

        Ok(SearchTextOutput { matches })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileInput {
    #[schemars(description = "Path to the file to read as UTF-8 text.")]
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileOutput {
    #[schemars(description = "Full UTF-8 contents of the requested file.")]
    pub contents: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReadFileTool;

impl ReadFileTool {
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    type Input = ReadFileInput;
    type Output = ReadFileOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolMetadata::new("read_file", "Reads a file as UTF-8 text")
            .descriptor_for::<ReadFileInput, ReadFileOutput>()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        let path = require_file(&input.path)?;
        let contents = fs::read_to_string(path).map_err(io_error)?;

        Ok(ReadFileOutput { contents })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadRangeInput {
    #[schemars(description = "Path to the text file to read from.")]
    pub path: PathBuf,
    #[schemars(description = "1-based first line number to include.")]
    pub start_line: usize,
    #[schemars(description = "1-based last line number to include, inclusive.")]
    pub end_line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadRangeLine {
    #[schemars(description = "1-based line number in the source file.")]
    pub number: usize,
    #[schemars(description = "Text content for the returned line.")]
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadRangeOutput {
    #[schemars(description = "Returned lines within the requested inclusive range.")]
    pub lines: Vec<ReadRangeLine>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReadRangeTool;

impl ReadRangeTool {
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ReadRangeTool {
    type Input = ReadRangeInput;
    type Output = ReadRangeOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolMetadata::new(
            "read_range",
            "Reads an inclusive line range from a text file",
        )
        .descriptor_for::<ReadRangeInput, ReadRangeOutput>()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        if input.start_line == 0 {
            return Err(ToolError::InvalidInput(String::from(
                "start_line must be greater than zero",
            )));
        }
        if input.end_line < input.start_line {
            return Err(ToolError::InvalidInput(String::from(
                "end_line must be greater than or equal to start_line",
            )));
        }

        let path = require_file(&input.path)?;
        let contents = fs::read_to_string(path).map_err(io_error)?;
        let lines = contents
            .lines()
            .enumerate()
            .filter_map(|(index, line)| {
                let number = index + 1;
                (input.start_line..=input.end_line)
                    .contains(&number)
                    .then(|| ReadRangeLine {
                        number,
                        text: line.to_string(),
                    })
            })
            .collect();

        Ok(ReadRangeOutput { lines })
    }
}

fn require_directory(path: &Path) -> Result<&Path, ToolError> {
    if path.is_dir() {
        Ok(path)
    } else {
        Err(ToolError::InvalidInput(format!(
            "directory does not exist: {}",
            path.display()
        )))
    }
}

fn walkdir_error(error: walkdir::Error) -> ToolError {
    let path = error
        .path()
        .map(|path| format!(" '{}'", path.display()))
        .unwrap_or_default();
    ToolError::ExecutionFailed(format!("failed to walk directory{path}: {error}"))
}

fn require_file(path: &Path) -> Result<&Path, ToolError> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(ToolError::InvalidInput(format!(
            "file does not exist: {}",
            path.display()
        )))
    }
}

fn relative_path(root: &Path, path: &Path) -> Result<String, ToolError> {
    let relative = path.strip_prefix(root).map_err(|error| {
        ToolError::ExecutionFailed(format!(
            "failed to compute relative path for {}: {error}",
            path.display()
        ))
    })?;

    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn io_error(error: std::io::Error) -> ToolError {
    ToolError::ExecutionFailed(error.to_string())
}
