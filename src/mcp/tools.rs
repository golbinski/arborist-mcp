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

// ── export_graph ──────────────────────────────────────────────────────────────

pub fn export_graph(ctx: &ToolContext, params: &Value) -> Result<Value> {
    let project = params["project"].as_str().context("missing project")?;
    let output_path = params["output_path"].as_str().context("missing output_path")?;
    let root_node = params["root_node"].as_str();
    let label = params["label"].as_str().and_then(NodeLabel::from_str);
    let depth = params["depth"].as_u64().unwrap_or(3).min(8) as usize;
    let max_nodes = params["max_nodes"].as_u64().unwrap_or(200).clamp(10, 500) as usize;

    let store = ctx.open(project)?;
    let locked = store.lock().unwrap();

    let subgraph = extract_subgraph(&locked, project, root_node, label, depth, max_nodes)?;
    drop(locked);

    let node_count = subgraph.nodes.len();
    let edge_count = subgraph.edges.len();

    let html = render_html(&subgraph, project);

    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    std::fs::write(output_path, html.as_bytes())
        .with_context(|| format!("writing HTML to {}", output_path))?;

    Ok(json!({
        "output_path": output_path,
        "nodes": node_count,
        "edges": edge_count,
        "message": format!("Graph written to {}. Open in a browser (requires internet for CDN scripts).", output_path),
    }))
}

// ── Subgraph extraction ───────────────────────────────────────────────────────

struct SubGraph {
    nodes: Vec<crate::graph::schema::Node>,
    /// (source_id, target_id, edge_type)
    edges: Vec<(i64, i64, EdgeType)>,
}

fn extract_subgraph(
    locked: &GraphStore,
    project: &str,
    root_node: Option<&str>,
    label: Option<NodeLabel>,
    depth: usize,
    max_nodes: usize,
) -> Result<SubGraph> {
    use crate::graph::store::NeighborDirection;

    if let Some(root_qname) = root_node {
        // Mode A: BFS from root node
        let roots = locked.search_nodes(
            project,
            &format!("^{}$", regex::escape(root_qname)),
            None,
            1,
        )?;
        let root = roots.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("root node '{}' not found", root_qname))?;

        let mut visited: HashSet<i64> = HashSet::new();
        let mut queue: std::collections::VecDeque<(i64, usize)> = std::collections::VecDeque::new();
        let mut edge_set: Vec<(i64, i64, EdgeType)> = Vec::new();

        visited.insert(root.id);
        queue.push_back((root.id, 0));

        while let Some((node_id, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }
            let neighbors = locked.get_neighbors(node_id, None, NeighborDirection::Both)?;
            for (nb_id, et, _) in neighbors {
                // Record the edge regardless of label filter
                edge_set.push((node_id, nb_id, et));

                if visited.contains(&nb_id) || visited.len() >= max_nodes {
                    continue;
                }
                // Apply label filter: only add the neighbor node to the subgraph
                // if it matches (but still traverse through it regardless)
                if let Some(filter_label) = label {
                    if let Ok(Some(nb)) = locked.get_node_by_id(nb_id) {
                        if nb.label != filter_label {
                            visited.insert(nb_id); // mark visited so we don't re-visit
                            continue;
                        }
                    }
                }
                visited.insert(nb_id);
                queue.push_back((nb_id, current_depth + 1));
            }
        }

        // Fetch all visited nodes
        let mut nodes = Vec::with_capacity(visited.len());
        for id in &visited {
            if let Ok(Some(n)) = locked.get_node_by_id(*id) {
                if label.map(|l| n.label == l).unwrap_or(true) {
                    nodes.push(n);
                }
            }
        }

        // Keep only edges where both endpoints are in our final node set
        let node_id_set: HashSet<i64> = nodes.iter().map(|n| n.id).collect();
        let edges = edge_set
            .into_iter()
            .filter(|(s, t, _)| node_id_set.contains(s) && node_id_set.contains(t))
            .collect::<HashSet<_>>() // dedup
            .into_iter()
            .collect();

        Ok(SubGraph { nodes, edges })
    } else {
        // Mode B: top-N highest-degree nodes as seeds, plus their outbound neighbors
        let seeds = locked.get_high_degree_nodes(project, label, max_nodes / 2)?;
        let mut node_ids: HashSet<i64> = seeds.iter().map(|n| n.id).collect();

        // Expand one hop outbound from each seed
        for seed in &seeds {
            if node_ids.len() >= max_nodes {
                break;
            }
            let neighbors =
                locked.get_neighbors(seed.id, None, NeighborDirection::Outbound)?;
            for (nb_id, _, _) in neighbors {
                if node_ids.len() >= max_nodes {
                    break;
                }
                node_ids.insert(nb_id);
            }
        }

        // Fetch full Node structs for all ids
        let id_vec: Vec<i64> = node_ids.into_iter().collect();
        let mut nodes = Vec::with_capacity(id_vec.len());
        for id in &id_vec {
            if let Ok(Some(n)) = locked.get_node_by_id(*id) {
                if label.map(|l| n.label == l).unwrap_or(true) {
                    nodes.push(n);
                }
            }
        }

        // Fetch edges between nodes in the subgraph
        let final_ids: Vec<i64> = nodes.iter().map(|n| n.id).collect();
        let raw_edges = locked.get_edges_between(&final_ids)?;
        let edges = raw_edges
            .into_iter()
            .map(|e| (e.source_id, e.target_id, e.edge_type))
            .collect();

        Ok(SubGraph { nodes, edges })
    }
}

// ── HTML rendering ────────────────────────────────────────────────────────────

fn render_html(subgraph: &SubGraph, project: &str) -> String {
    let cy_nodes: Vec<serde_json::Value> = subgraph.nodes.iter().map(|n| {
        json!({
            "id": n.id.to_string(),
            "label": n.label.as_str(),
            "qname": n.qualified_name,
            "file": n.file_path.as_deref().unwrap_or(""),
            "line": n.line_start.unwrap_or(0),
        })
    }).collect();

    let cy_edges: Vec<serde_json::Value> = subgraph.edges.iter().map(|(src, tgt, et)| {
        json!({
            "source": src.to_string(),
            "target": tgt.to_string(),
            "etype": et.as_str(),
        })
    }).collect();

    let graph_data = json!({
        "project": project,
        "nodes": cy_nodes,
        "edges": cy_edges,
    });

    HTML_TEMPLATE.replace("{GRAPH_DATA}", &graph_data.to_string())
}

// ── HTML template ─────────────────────────────────────────────────────────────

const HTML_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>arborist — {TITLE}</title>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { display: flex; height: 100vh; background: #1e1e2e; color: #cdd6f4;
           font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace; font-size: 12px; }
    #sidebar {
      width: 280px; min-width: 220px; display: flex; flex-direction: column;
      background: #181825; border-right: 1px solid #313244; overflow: hidden;
    }
    #sidebar-header {
      padding: 12px 14px 8px; border-bottom: 1px solid #313244; flex-shrink: 0;
    }
    #sidebar-header h1 { font-size: 13px; font-weight: 700; color: #89b4fa; letter-spacing: 0.05em; }
    #sidebar-header p  { font-size: 10px; color: #6c7086; margin-top: 2px; }
    #search-wrap { padding: 8px 14px; border-bottom: 1px solid #313244; flex-shrink: 0; }
    #search { width: 100%; padding: 5px 8px; background: #313244; border: 1px solid #45475a;
              border-radius: 4px; color: #cdd6f4; font-size: 11px; font-family: inherit; outline: none; }
    #search:focus { border-color: #89b4fa; }
    #sidebar-body { flex: 1; overflow-y: auto; padding: 10px 14px; }
    #node-info { margin-bottom: 12px; }
    #node-info .ni-qname { color: #cdd6f4; word-break: break-all; font-size: 11px; line-height: 1.5; }
    #node-info .ni-tag { display: inline-block; padding: 1px 6px; border-radius: 3px;
                         font-size: 10px; font-weight: 600; margin: 4px 0; }
    #node-info .ni-file { color: #6c7086; font-size: 10px; word-break: break-all; }
    #node-info .ni-empty { color: #45475a; font-size: 11px; }
    .section-title { font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
                     color: #6c7086; text-transform: uppercase; margin-bottom: 6px; margin-top: 10px; }
    .legend-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .legend-line { width: 18px; height: 2px; flex-shrink: 0; }
    .legend-label { color: #a6adc8; font-size: 10px; }
    .edge-toggle { display: flex; align-items: center; gap: 6px; margin: 3px 0; cursor: pointer; }
    .edge-toggle input { cursor: pointer; accent-color: #89b4fa; }
    #cy { flex: 1; }
    #tooltip {
      position: fixed; display: none; background: #313244; border: 1px solid #45475a;
      border-radius: 6px; padding: 8px 10px; pointer-events: none; z-index: 10;
      max-width: 340px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    #tooltip .tt-qname { color: #cdd6f4; font-size: 11px; word-break: break-all; font-weight: 600; }
    #tooltip .tt-meta  { color: #a6adc8; font-size: 10px; margin-top: 3px; }
    ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #313244; border-radius: 2px; }
  </style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <h1>arborist-mcp</h1>
    <p id="graph-meta"></p>
  </div>
  <div id="search-wrap">
    <input id="search" type="search" placeholder="Search nodes…" autocomplete="off">
  </div>
  <div id="sidebar-body">
    <div id="node-info"><p class="ni-empty">Click a node to inspect it.</p></div>
    <div class="section-title">Node types</div>
    <div id="node-legend"></div>
    <div class="section-title" style="margin-top:14px">Edge types</div>
    <div id="edge-legend"></div>
  </div>
</div>
<div id="cy"></div>
<div id="tooltip"></div>

<script>
const GRAPH_DATA = {GRAPH_DATA};

const NODE_COLORS = {
  Project:'#89b4fa', File:'#a6e3a1', Namespace:'#cba6f7',
  Class:'#fab387', Struct:'#f38ba8', Function:'#94e2d5',
  Method:'#89dceb', Template:'#f9e2af', Enum:'#eba0ac', Variable:'#b4befe'
};
const EDGE_COLORS = {
  CALLS:'#94e2d5', INHERITS:'#f38ba8', OVERRIDES:'#fab387',
  INSTANTIATES:'#cba6f7', USES_TYPE:'#b4befe',
  DEFINES:'#45475a', CONTAINS:'#383845', INCLUDES:'#313244'
};
const STRUCTURAL_EDGES = new Set(['DEFINES','CONTAINS','INCLUDES']);

// ── Sidebar meta ──────────────────────────────────────────────────────────────
document.getElementById('graph-meta').textContent =
  `${GRAPH_DATA.project} · ${GRAPH_DATA.nodes.length} nodes · ${GRAPH_DATA.edges.length} edges`;

// ── Build element arrays ──────────────────────────────────────────────────────
const allEdgesByType = {};
GRAPH_DATA.edges.forEach(e => {
  (allEdgesByType[e.etype] = allEdgesByType[e.etype] || []).push(e);
});

function makeElements(hiddenTypes) {
  const nodes = GRAPH_DATA.nodes.map(n => ({
    group: 'nodes',
    data: { id: n.id, label: n.label, qname: n.qname, file: n.file, line: n.line }
  }));
  const edges = GRAPH_DATA.edges
    .filter(e => !hiddenTypes.has(e.etype))
    .map(e => ({
      group: 'edges',
      data: { id: `${e.source}-${e.target}-${e.etype}`, source: e.source, target: e.target, etype: e.etype }
    }));
  return [...nodes, ...edges];
}

const hiddenEdgeTypes = new Set(STRUCTURAL_EDGES);
const cyStyle = [
  {
    selector: 'node',
    style: {
      'background-color': ele => NODE_COLORS[ele.data('label')] || '#7f849c',
      'label': ele => {
        const q = ele.data('qname');
        const short = q.split('::').pop();
        return short.length > 28 ? short.slice(0, 26) + '…' : short;
      },
      'font-size': '9px',
      'font-family': 'ui-monospace, monospace',
      'color': '#cdd6f4',
      'text-valign': 'bottom',
      'text-margin-y': '4px',
      'text-outline-width': '2px',
      'text-outline-color': '#1e1e2e',
      'width': '20px', 'height': '20px',
      'border-width': '0px',
    }
  },
  {
    selector: 'node:selected, node.highlighted',
    style: { 'border-width': '2.5px', 'border-color': '#f5c2e7', 'width': '24px', 'height': '24px' }
  },
  {
    selector: 'node.dimmed',
    style: { 'opacity': 0.12 }
  },
  {
    selector: 'edge',
    style: {
      'width': 1.2,
      'line-color': ele => EDGE_COLORS[ele.data('etype')] || '#585b70',
      'target-arrow-color': ele => EDGE_COLORS[ele.data('etype')] || '#585b70',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 0.7,
      'curve-style': 'bezier',
      'opacity': 0.65,
    }
  },
  {
    selector: 'edge.highlighted',
    style: { 'opacity': 1, 'width': 2 }
  },
  {
    selector: 'edge.dimmed',
    style: { 'opacity': 0.05 }
  },
];

const layoutName = GRAPH_DATA.nodes.length > 300 ? 'cose' : 'dagre';
const layoutOpts = layoutName === 'dagre'
  ? { name: 'dagre', rankDir: 'TB', nodeSep: 40, rankSep: 70, padding: 30, animate: false }
  : { name: 'cose', nodeRepulsion: 8000, idealEdgeLength: 80, animate: false, randomize: true };

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: makeElements(hiddenEdgeTypes),
  style: cyStyle,
  layout: layoutOpts,
  minZoom: 0.05, maxZoom: 4,
});

// ── Interactions ──────────────────────────────────────────────────────────────
cy.on('tap', 'node', function(evt) {
  const node = evt.target;
  const neighborhood = node.closedNeighborhood();
  cy.elements().removeClass('highlighted dimmed');
  neighborhood.addClass('highlighted');
  cy.elements().not(neighborhood).addClass('dimmed');
  showNodeInfo(node.data());
});

cy.on('tap', function(evt) {
  if (evt.target === cy) {
    cy.elements().removeClass('highlighted dimmed');
    document.getElementById('node-info').innerHTML =
      '<p class="ni-empty">Click a node to inspect it.</p>';
  }
});

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tooltip = document.getElementById('tooltip');
cy.on('mouseover', 'node', function(evt) {
  const d = evt.target.data();
  tooltip.innerHTML =
    `<div class="tt-qname">${d.qname}</div>` +
    `<div class="tt-meta">${d.label}${d.file ? ' · ' + d.file.split('/').pop() + ':' + d.line : ''}</div>`;
  tooltip.style.display = 'block';
});
cy.on('mousemove', 'node', function(evt) {
  tooltip.style.left = (evt.originalEvent.clientX + 14) + 'px';
  tooltip.style.top  = (evt.originalEvent.clientY - 10) + 'px';
});
cy.on('mouseout', 'node', () => { tooltip.style.display = 'none'; });

// ── Search ────────────────────────────────────────────────────────────────────
document.getElementById('search').addEventListener('input', function() {
  const q = this.value.trim().toLowerCase();
  if (!q) { cy.nodes().style('opacity', 1); return; }
  cy.nodes().forEach(n => {
    n.style('opacity', n.data('qname').toLowerCase().includes(q) ? 1 : 0.08);
  });
});

// ── Node info panel ───────────────────────────────────────────────────────────
function showNodeInfo(d) {
  const color = NODE_COLORS[d.label] || '#7f849c';
  document.getElementById('node-info').innerHTML =
    `<div class="ni-qname">${d.qname}</div>` +
    `<span class="ni-tag" style="background:${color}22;color:${color}">${d.label}</span><br>` +
    (d.file ? `<div class="ni-file">${d.file}:${d.line}</div>` : '');
}

// ── Node legend ───────────────────────────────────────────────────────────────
const presentLabels = [...new Set(GRAPH_DATA.nodes.map(n => n.label))].sort();
const nodeLegend = document.getElementById('node-legend');
presentLabels.forEach(lbl => {
  const row = document.createElement('div');
  row.className = 'legend-row';
  row.innerHTML =
    `<div class="legend-dot" style="background:${NODE_COLORS[lbl]||'#7f849c'}"></div>` +
    `<span class="legend-label">${lbl}</span>`;
  nodeLegend.appendChild(row);
});

// ── Edge type toggles ─────────────────────────────────────────────────────────
const presentEtypes = [...new Set(GRAPH_DATA.edges.map(e => e.etype))].sort();
const edgeLegend = document.getElementById('edge-legend');
presentEtypes.forEach(etype => {
  const color = EDGE_COLORS[etype] || '#585b70';
  const checked = !STRUCTURAL_EDGES.has(etype);
  const row = document.createElement('label');
  row.className = 'edge-toggle';
  row.innerHTML =
    `<input type="checkbox" data-etype="${etype}" ${checked ? 'checked' : ''}>` +
    `<div class="legend-line" style="background:${color}"></div>` +
    `<span class="legend-label">${etype}</span>`;
  edgeLegend.appendChild(row);

  row.querySelector('input').addEventListener('change', function() {
    if (this.checked) {
      hiddenEdgeTypes.delete(etype);
      // Re-add edges of this type
      const toAdd = (allEdgesByType[etype] || []).map(e => ({
        group: 'edges',
        data: { id: `${e.source}-${e.target}-${e.etype}`, source: e.source, target: e.target, etype: e.etype }
      }));
      cy.add(toAdd);
    } else {
      hiddenEdgeTypes.add(etype);
      cy.remove(`edge[etype="${etype}"]`);
    }
    cy.layout(layoutOpts).run();
  });
});
</script>
</body>
</html>"#;
