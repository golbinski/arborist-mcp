/// Tool handler implementations.
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::embeddings;
use crate::graph::schema::{EdgeType, NodeLabel};
use crate::graph::store::{GraphStore, NeighborDirection};
use crate::pipeline::{self, IndexingConfig};

pub struct ToolContext {
    pub db_dir: PathBuf,
}

impl ToolContext {
    fn db_path(&self, project: &str) -> PathBuf {
        self.db_dir.join(format!("{}.db", sanitize_name(project)))
    }

    fn open(&self, project: &str) -> Result<Arc<Mutex<GraphStore>>> {
        let path = self.db_path(project);
        crate::graph::open_store(&path)
    }

    fn open_shared(&self) -> Result<Arc<Mutex<GraphStore>>> {
        let path = self.db_dir.join("arborist.db");
        crate::graph::open_store(&path)
    }
}

fn sanitize_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

// ── Tool handlers ─────────────────────────────────────────────────────────────

pub fn index_repository(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let repo_path = params["repo_path"]
        .as_str()
        .context("missing repo_path")?
        .to_owned();

    let compile_commands_path = params["compile_commands_path"]
        .as_str()
        .map(PathBuf::from);

    // Derive project name from last directory component
    let project_name = Path::new(&repo_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed".to_owned());

    let db_path = ctx.db_path(&project_name);

    // Run in background thread so MCP response returns promptly
    let config = IndexingConfig {
        repo_path: PathBuf::from(&repo_path),
        project_name: project_name.clone(),
        compile_commands_path,
        db_path,
    };

    std::thread::spawn(move || {
        if let Err(e) = pipeline::run(config) {
            tracing::error!("indexing failed: {}", e);
        }
    });

    Ok(json!({
        "status": "started",
        "project": project_name,
        "message": format!("Indexing {} in background. Use index_status to check progress.", repo_path)
    }))
}

pub fn list_projects(ctx: &ToolContext, _params: &Value) -> Result<Value> {
    // Enumerate all .db files in db_dir
    let mut projects: Vec<Value> = Vec::new();

    let entries = match std::fs::read_dir(&ctx.db_dir) {
        Ok(e) => e,
        Err(_) => return Ok(json!({ "projects": [] })),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("db") {
            if let Ok(store) = crate::graph::open_store(&path) {
                if let Ok(rows) = store.lock().unwrap().list_projects() {
                    projects.extend(rows);
                }
            }
        }
    }

    Ok(json!({ "projects": projects }))
}

pub fn index_status(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let store = ctx.open(project)?;
    let status = store.lock().unwrap().get_project_status(project)?;
    Ok(status.unwrap_or(json!({ "error": "project not found" })))
}

pub fn search_graph(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let pattern = params["name_pattern"]
        .as_str()
        .context("missing name_pattern")?;
    let label = params["label"]
        .as_str()
        .and_then(NodeLabel::from_str);
    let limit = params["limit"].as_u64().unwrap_or(20) as usize;

    let store = ctx.open(project)?;
    let nodes = store.lock().unwrap().search_nodes(project, pattern, label, limit)?;

    let results: Vec<Value> = nodes
        .into_iter()
        .map(|n| json!({
            "id": n.id,
            "label": n.label.as_str(),
            "qualified_name": n.qualified_name,
            "file_path": n.file_path,
            "line_start": n.line_start,
            "line_end": n.line_end,
        }))
        .collect();

    let count = results.len();
    Ok(json!({ "results": results, "count": count }))
}

pub fn trace_calls(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let function_name = params["function_name"]
        .as_str()
        .context("missing function_name")?;
    let direction = params["direction"]
        .as_str()
        .and_then(NeighborDirection::from_str)
        .unwrap_or(NeighborDirection::Outbound);
    let depth = params["depth"].as_u64().unwrap_or(3) as usize;

    let store = ctx.open(project)?;
    let locked = store.lock().unwrap();

    // Find root node
    let roots = locked.search_nodes(project, function_name, None, 5)?;
    if roots.is_empty() {
        return Ok(json!({ "error": format!("no node matching '{}'", function_name) }));
    }
    let root = &roots[0];

    // BFS
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<(i64, usize)> = VecDeque::new();
    let mut trace: Vec<Value> = Vec::new();

    queue.push_back((root.id, 0));
    visited.insert(root.id);

    while let Some((node_id, level)) = queue.pop_front() {
        let neighbors = locked.get_neighbors(node_id, Some(EdgeType::Calls), direction)?;
        for (nb_id, _et, _) in neighbors {
            if !visited.contains(&nb_id) {
                visited.insert(nb_id);
                if let Ok(Some(nb_node)) = locked.get_node_by_id(nb_id) {
                    trace.push(json!({
                        "from_id": node_id,
                        "to_id": nb_id,
                        "qualified_name": nb_node.qualified_name,
                        "label": nb_node.label.as_str(),
                        "file_path": nb_node.file_path,
                        "depth": level + 1,
                    }));
                }
                if level + 1 < depth {
                    queue.push_back((nb_id, level + 1));
                }
            }
        }
    }

    Ok(json!({
        "root": {
            "id": root.id,
            "qualified_name": root.qualified_name,
            "label": root.label.as_str(),
        },
        "trace": trace,
        "direction": params["direction"].as_str().unwrap_or("outbound"),
    }))
}

pub fn find_similar(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let function_name = params["function_name"]
        .as_str()
        .context("missing function_name")?;
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;

    let store = ctx.open(project)?;
    let anchor_nodes = store
        .lock()
        .unwrap()
        .search_nodes(project, function_name, None, 1)?;

    if anchor_nodes.is_empty() {
        return Ok(json!({ "error": format!("no node matching '{}'", function_name) }));
    }

    let anchor = &anchor_nodes[0];
    let similar = embeddings::find_similar(store.clone(), project, anchor.id, top_k)?;

    let locked = store.lock().unwrap();
    let results: Vec<Value> = similar
        .into_iter()
        .filter_map(|(id, score)| {
            locked.get_node_by_id(id).ok().flatten().map(|n| {
                json!({
                    "id": n.id,
                    "qualified_name": n.qualified_name,
                    "label": n.label.as_str(),
                    "file_path": n.file_path,
                    "similarity": score,
                })
            })
        })
        .collect();

    Ok(json!({
        "anchor": {
            "id": anchor.id,
            "qualified_name": anchor.qualified_name,
        },
        "similar": results,
    }))
}

pub fn get_snippet(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let qualified_name = params["qualified_name"]
        .as_str()
        .context("missing qualified_name")?;

    let store = ctx.open(project)?;
    let nodes = store
        .lock()
        .unwrap()
        .search_nodes(project, &format!("^{}$", regex::escape(qualified_name)), None, 1)?;

    if nodes.is_empty() {
        return Ok(json!({ "error": format!("symbol '{}' not found", qualified_name) }));
    }

    let node = &nodes[0];
    let file_path = match &node.file_path {
        Some(p) => p.clone(),
        None => return Ok(json!({ "error": "symbol has no file path" })),
    };

    let source = std::fs::read_to_string(&file_path)
        .with_context(|| format!("reading {}", file_path))?;

    let start = node.line_start.unwrap_or(1) as usize;
    let end = node.line_end.unwrap_or(start as i32) as usize;

    let snippet: String = source
        .lines()
        .enumerate()
        .skip(start.saturating_sub(1))
        .take(end - start + 1)
        .map(|(i, line)| format!("{:4} | {}", i + 1, line))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(json!({
        "qualified_name": node.qualified_name,
        "file_path": file_path,
        "line_start": start,
        "line_end": end,
        "snippet": snippet,
    }))
}

pub fn detect_changes(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;

    let store = ctx.open(project)?;
    let locked = store.lock().unwrap();

    // Get repo path from project record
    let status = locked.get_project_status(project)?
        .ok_or_else(|| anyhow::anyhow!("project not found"))?;
    let repo_path = status["repo_path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("project has no repo_path"))?
        .to_owned();

    drop(locked);

    // Run git diff --name-only
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", "HEAD"])
        .current_dir(&repo_path)
        .output()
        .context("running git diff")?;

    let changed_files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| {
            let full = PathBuf::from(&repo_path).join(l);
            full.to_string_lossy().into_owned()
        })
        .collect();

    // Find nodes in those files
    let locked = store.lock().unwrap();
    let mut affected: Vec<Value> = Vec::new();

    for file_path in &changed_files {
        let file_qname = format!("file:{}", file_path);
        if let Ok(Some(file_id)) = locked.get_node_id(project, &file_qname) {
            // DEFINES edges from this file
            let defined = locked.get_neighbors(file_id, Some(EdgeType::Defines), NeighborDirection::Outbound)?;
            for (sym_id, _, _) in defined {
                if let Ok(Some(sym)) = locked.get_node_by_id(sym_id) {
                    affected.push(json!({
                        "qualified_name": sym.qualified_name,
                        "label": sym.label.as_str(),
                        "file_path": sym.file_path,
                        "change_type": "modified",
                    }));
                }
            }
        }
    }

    let affected_count = affected.len();
    Ok(json!({
        "changed_files": changed_files,
        "affected_symbols": affected,
        "count": affected_count,
    }))
}

pub fn query_graph(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let query = params["query"].as_str().context("missing query")?;

    // Minimal query language:
    // MATCH (n:Label {name: "pattern"}) RETURN n
    // MATCH (n)-[r:TYPE]->(m) WHERE n.name = "pattern" RETURN n, r, m LIMIT k

    parse_and_execute_query(ctx, project, query)
}

fn parse_and_execute_query(ctx: &ToolContext, project: &str, query: &str) -> Result<Value> {
    let store = ctx.open(project)?;
    let locked = store.lock().unwrap();

    let query_upper = query.to_uppercase();

    // Extract LIMIT
    let limit: usize = if let Some(pos) = query_upper.find("LIMIT") {
        let rest = query[pos + 5..].trim();
        rest.split_whitespace()
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(20)
    } else {
        20
    };

    // Extract label filter  e.g. (n:Function)
    let label = {
        let re = regex::Regex::new(r"\(n:(\w+)").unwrap();
        re.captures(query)
            .and_then(|c| c.get(1))
            .and_then(|m| NodeLabel::from_str(m.as_str()))
    };

    // Extract name pattern from {name: "..."} or WHERE n.name = "..."
    let name_pattern = {
        let re = regex::Regex::new(r#"(?:name:\s*["']([^"']+)["']|n\.(?:name|qualified_name)\s*=\s*["']([^"']+)["'])"#).unwrap();
        re.captures(query)
            .and_then(|c| c.get(1).or_else(|| c.get(2)))
            .map(|m| m.as_str().to_owned())
            .unwrap_or_else(|| ".*".to_owned())
    };

    let nodes = locked.search_nodes(project, &name_pattern, label, limit)?;

    // If query has relationship pattern, fetch edges too
    let include_edges = query_upper.contains("->") || query_upper.contains("<-") || query_upper.contains("-[");

    let result_nodes: Vec<Value> = nodes
        .iter()
        .map(|n| json!({
            "id": n.id,
            "label": n.label.as_str(),
            "qualified_name": n.qualified_name,
            "file_path": n.file_path,
            "line_start": n.line_start,
        }))
        .collect();

    let node_count = result_nodes.len();
    let mut result = json!({
        "nodes": result_nodes,
        "count": node_count,
    });

    if include_edges {
        let mut all_edges: Vec<Value> = Vec::new();
        for node in &nodes {
            let edges = locked.get_neighbors(node.id, None, NeighborDirection::Outbound)?;
            for (tgt_id, et, _) in edges {
                all_edges.push(json!({
                    "from": node.id,
                    "to": tgt_id,
                    "type": et.as_str(),
                }));
            }
        }
        result["edges"] = json!(all_edges);
    }

    Ok(result)
}

pub fn delete_project(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let store = ctx.open(project)?;
    let n = store.lock().unwrap().delete_project(project)?;

    // Also remove the db file if it's now empty
    let db_path = ctx.db_path(project);
    let _ = std::fs::remove_file(&db_path);

    Ok(json!({
        "deleted": true,
        "project": project,
        "nodes_removed": n,
    }))
}
