/// Multi-pass indexing pipeline (orchestrated with rayon).
use anyhow::Result;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::embeddings;
use crate::graph::schema::{EdgeType, NodeLabel};
use crate::graph::store::GraphStore;
use crate::graph::GraphBuffer;
use crate::parser::libclang::LibclangResolver;
use crate::parser::{
    collect_cpp_files, ingest_call_sites, ingest_file_result, ingest_includes, parse_file,
};

pub struct IndexingConfig {
    pub repo_path: PathBuf,
    pub project_name: String,
    pub compile_commands_path: Option<PathBuf>,
    pub db_path: PathBuf,
}

/// Run the full indexing pipeline.
///
/// All writes go into an in-memory SQLite database for maximum throughput
/// (no fsync per transaction).  When indexing is complete the database is
/// persisted to `config.db_path` in a single bulk backup operation.
pub fn run(config: IndexingConfig) -> Result<()> {
    let store = Arc::new(Mutex::new(GraphStore::open_memory()?));

    {
        let locked = store.lock().unwrap();
        locked.upsert_project(&config.project_name, &config.repo_path.to_string_lossy())?;
    }

    let result = run_inner(&config, store.clone());

    {
        let locked = store.lock().unwrap();
        match &result {
            Ok(()) => locked.set_project_done(&config.project_name)?,
            Err(e) => locked.set_project_error(&config.project_name, &e.to_string())?,
        }
    }

    // Persist to disk regardless of success or failure so index_status reflects the error.
    tracing::info!("persisting graph to {}", config.db_path.display());
    store.lock().unwrap().backup_to_file(&config.db_path)?;

    result
}

fn run_inner(config: &IndexingConfig, store: Arc<Mutex<GraphStore>>) -> Result<()> {
    // ── Pass 1: Discover & parse files (parallel) ─────────────────────────
    tracing::info!("discovering C++ files in {}", config.repo_path.display());
    let files = collect_cpp_files(&config.repo_path);
    tracing::info!("found {} C++ files", files.len());

    let file_results: Vec<_> = files
        .par_iter()
        .filter_map(|path| match parse_file(path) {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::warn!("parse error for {}: {}", path.display(), e);
                None
            }
        })
        .collect();

    // ── Pass 2: Ensure a Project node ────────────────────────────────────
    let mut graph = GraphBuffer::new(store.clone());
    let _project_id = graph.ensure_node(
        &config.project_name,
        NodeLabel::Project,
        &config.project_name,
        None,
        None,
        None,
    )?;

    // ── Pass 3: Ingest symbols + File nodes ──────────────────────────────
    tracing::info!("ingesting symbols from {} files", file_results.len());
    for result in &file_results {
        ingest_file_result(result, &config.project_name, &mut graph)?;
    }

    // ── Pass 4: Wire call sites ───────────────────────────────────────────
    tracing::info!("resolving call sites");
    let all_calls: Vec<_> = file_results
        .iter()
        .flat_map(|r| r.calls.iter().cloned())
        .collect();
    ingest_call_sites(&all_calls, &config.project_name, &mut graph)?;

    // ── Pass 5: Wire include edges ────────────────────────────────────────
    let all_includes: Vec<_> = file_results
        .iter()
        .flat_map(|r| r.includes.iter().cloned())
        .collect();
    ingest_includes(&all_includes, &config.project_name, &mut graph)?;

    // ── Pass 6: libclang supplementary resolution (optional) ─────────────
    let cc_path = config
        .compile_commands_path
        .clone()
        .unwrap_or_else(|| config.repo_path.join("compile_commands.json"));

    if let Some(resolver) = LibclangResolver::try_new(&cc_path) {
        tracing::info!("libclang resolver active");
        for result in &file_results {
            let (extra_calls, inherits, usages) = resolver.resolve_file(&result.path);

            for rc in &extra_calls {
                let src = store
                    .lock()
                    .unwrap()
                    .get_node_id(&config.project_name, &rc.caller_qname)?;
                let tgt = store
                    .lock()
                    .unwrap()
                    .get_node_id(&config.project_name, &rc.callee_qname)?;
                if let (Some(s), Some(t)) = (src, tgt) {
                    graph.upsert_edge(s, t, EdgeType::Calls)?;
                }
            }

            for ri in &inherits {
                // Ensure both class nodes exist (they may not have been extracted if in a system header)
                let derived_id = graph.ensure_node(
                    &config.project_name,
                    NodeLabel::Class,
                    &ri.derived_qname,
                    None,
                    None,
                    None,
                )?;
                let base_id = graph.ensure_node(
                    &config.project_name,
                    NodeLabel::Class,
                    &ri.base_qname,
                    None,
                    None,
                    None,
                )?;
                graph.upsert_edge(derived_id, base_id, EdgeType::Inherits)?;
            }

            for ru in &usages {
                let user = store
                    .lock()
                    .unwrap()
                    .get_node_id(&config.project_name, &ru.user_qname)?;
                let typ = store
                    .lock()
                    .unwrap()
                    .get_node_id(&config.project_name, &ru.type_qname)?;
                if let (Some(u), Some(t)) = (user, typ) {
                    graph.upsert_edge(u, t, EdgeType::UsesType)?;
                }
            }
        }
    } else {
        tracing::info!("libclang not available — skipping type-resolved pass");
    }

    // ── Pass 7: Cleora embeddings ─────────────────────────────────────────
    tracing::info!("computing Cleora embeddings");
    embeddings::compute_and_store(store.clone(), &config.project_name)?;

    tracing::info!("indexing complete for project '{}'", config.project_name);
    Ok(())
}
