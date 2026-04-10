pub mod libclang;
pub mod treesitter;

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::graph::{GraphBuffer};
use crate::graph::schema::{EdgeType, NodeLabel};
use treesitter::TreeSitterExtractor;

static CPP_EXTENSIONS: &[&str] = &[
    "cpp", "cxx", "cc", "c++", "C", "c",
    "hpp", "hxx", "hh", "h++", "h",
    "cu", "cuh", // CUDA
];

/// Returns true if `path` is a C++ source or header file.
pub fn is_cpp_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| CPP_EXTENSIONS.contains(&e))
        .unwrap_or(false)
}

/// Collect all C++ files under `repo_path`, skipping common build dirs.
pub fn collect_cpp_files(repo_path: &Path) -> Vec<PathBuf> {
    WalkDir::new(repo_path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "build" | "cmake-build-debug" | "cmake-build-release"
                | "_build" | "out" | "dist" | "target" | "vendor"
                | "third_party" | "3rdparty" | "extern" | "external"
            )
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_cpp_file(p))
        .collect()
}

/// Full single-file parse result.
pub struct FileResult {
    pub path: PathBuf,
    pub symbols: Vec<treesitter::Symbol>,
    pub calls: Vec<treesitter::CallSite>,
    pub includes: Vec<treesitter::IncludeDirective>,
}

/// Parse one file with tree-sitter.
pub fn parse_file(path: &Path) -> Result<FileResult> {
    let source = std::fs::read(path)?;
    let mut extractor = TreeSitterExtractor::new()?;
    let (symbols, calls, includes) = extractor.extract_file(path, &source)?;
    Ok(FileResult {
        path: path.to_owned(),
        symbols,
        calls,
        includes,
    })
}

/// Ingest a `FileResult` into the graph buffer.
///
/// Returns a mapping from simple callee names to qualified names for use
/// in the call-resolution phase.
pub fn ingest_file_result(
    result: &FileResult,
    project: &str,
    graph: &mut GraphBuffer,
) -> Result<()> {
    let file_path_str = result.path.to_string_lossy();

    // Ensure a File node exists
    let file_qname = format!("file:{}", file_path_str);
    let file_id = graph.ensure_node(
        project,
        NodeLabel::File,
        &file_qname,
        Some(&file_path_str),
        None,
        None,
    )?;

    for sym in &result.symbols {
        let sym_id = graph.ensure_node(
            project,
            sym.label,
            &sym.qualified_name,
            Some(&sym.file_path),
            Some(sym.line_start as i32),
            Some(sym.line_end as i32),
        )?;

        // File → DEFINES → Symbol
        graph.upsert_edge(file_id, sym_id, EdgeType::Defines)?;

        // Parent scope → CONTAINS → Symbol
        if let Some(ref parent) = sym.parent_scope {
            // Acquire and release the lock before calling upsert_edge (which also locks).
            let parent_id = graph
                .store()
                .lock()
                .unwrap()
                .get_node_id(project, parent)
                .ok()
                .flatten();
            if let Some(parent_id) = parent_id {
                graph.upsert_edge(parent_id, sym_id, EdgeType::Contains)?;
            }
        }
    }

    Ok(())
}

/// Second pass: wire up CALLS edges.
///
/// Builds an in-memory index (simple_name → node_id) once, then resolves each
/// call site with a HashMap lookup — O(nodes + calls) instead of O(calls × nodes).
pub fn ingest_call_sites(
    calls: &[treesitter::CallSite],
    project: &str,
    graph: &mut GraphBuffer,
) -> Result<()> {
    let store = graph.store();

    // Build index: last component of qualified_name → first matching node id.
    // Prefer exact matches (no "::") over qualified ones so simple names resolve
    // to free functions before member functions.
    let all_nodes = store.lock().unwrap().get_all_nodes(project)?;
    let mut name_index: HashMap<String, i64> = HashMap::new();
    for node in &all_nodes {
        let simple = node
            .qualified_name
            .rsplit("::")
            .next()
            .unwrap_or(&node.qualified_name)
            .to_owned();
        // Insert only if not already present (first-seen wins; exact entries inserted last below)
        name_index.entry(simple).or_insert(node.id);
    }
    // Also index full qualified names for callers
    let qname_index: HashMap<&str, i64> = all_nodes
        .iter()
        .map(|n| (n.qualified_name.as_str(), n.id))
        .collect();

    for call in calls {
        let caller_id = match qname_index.get(call.caller_qname.as_str()) {
            Some(&id) => id,
            None => continue,
        };
        if let Some(&callee_id) = name_index.get(&call.callee_name) {
            graph.upsert_edge(caller_id, callee_id, EdgeType::Calls)?;
        }
    }
    Ok(())
}

/// Third pass: wire up INCLUDES edges.
pub fn ingest_includes(
    includes: &[treesitter::IncludeDirective],
    project: &str,
    graph: &mut GraphBuffer,
) -> Result<()> {
    let store = graph.store();

    // Build index: file_path → node_id for all File nodes.
    let all_nodes = store.lock().unwrap().get_all_nodes(project)?;
    let file_index: HashMap<String, i64> = all_nodes
        .iter()
        .filter(|n| n.label == NodeLabel::File)
        .filter_map(|n| n.file_path.as_ref().map(|p| (p.clone(), n.id)))
        .collect();
    // Also map by qualified_name ("file:<path>")
    let qname_index: HashMap<&str, i64> = all_nodes
        .iter()
        .map(|n| (n.qualified_name.as_str(), n.id))
        .collect();

    for inc in includes {
        let from_qname = format!("file:{}", inc.from_file);
        let from_id = match qname_index.get(from_qname.as_str()) {
            Some(&id) => id,
            None => continue,
        };

        // Match included path by suffix against known file paths
        let to_id = file_index
            .iter()
            .find(|(fp, _)| fp.ends_with(inc.included_path.as_str()))
            .map(|(_, &id)| id);

        if let Some(to_id) = to_id {
            graph.upsert_edge(from_id, to_id, EdgeType::Includes)?;
        }
    }
    Ok(())
}
