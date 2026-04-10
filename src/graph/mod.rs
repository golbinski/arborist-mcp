pub mod schema;
pub mod store;

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use schema::{EdgeInput, EdgeType, NodeInput, NodeLabel};
use store::GraphStore;

/// In-memory buffer accumulated during a single indexing pass, then flushed.
pub struct GraphBuffer {
    store: Arc<Mutex<GraphStore>>,
    /// local name → node id cache to avoid repeated DB lookups
    id_cache: HashMap<(String, String), i64>,
}

impl GraphBuffer {
    pub fn new(store: Arc<Mutex<GraphStore>>) -> Self {
        Self {
            store,
            id_cache: HashMap::new(),
        }
    }

    pub fn upsert_node(&mut self, node: NodeInput) -> Result<i64> {
        let key = (node.project.clone(), node.qualified_name.clone());
        if let Some(&id) = self.id_cache.get(&key) {
            return Ok(id);
        }
        let id = self.store.lock().unwrap().upsert_node(&node)?;
        self.id_cache.insert(key, id);
        Ok(id)
    }

    pub fn ensure_node(
        &mut self,
        project: &str,
        label: NodeLabel,
        qualified_name: &str,
        file_path: Option<&str>,
        line_start: Option<i32>,
        line_end: Option<i32>,
    ) -> Result<i64> {
        self.upsert_node(NodeInput {
            project: project.to_owned(),
            label,
            qualified_name: qualified_name.to_owned(),
            file_path: file_path.map(|s| s.to_owned()),
            line_start,
            line_end,
            properties: serde_json::Value::Object(Default::default()),
        })
    }

    pub fn upsert_edge(
        &self,
        source_id: i64,
        target_id: i64,
        edge_type: EdgeType,
    ) -> Result<()> {
        self.store.lock().unwrap().upsert_edge(&EdgeInput {
            source_id,
            target_id,
            edge_type,
            properties: serde_json::Value::Object(Default::default()),
        })
    }

    pub fn store(&self) -> Arc<Mutex<GraphStore>> {
        self.store.clone()
    }
}

/// Opens (or creates) the graph store for a project's db file.
pub fn open_store(db_path: &Path) -> Result<Arc<Mutex<GraphStore>>> {
    let store = GraphStore::open(db_path)?;
    Ok(Arc::new(Mutex::new(store)))
}
