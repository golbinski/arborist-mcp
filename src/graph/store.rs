use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

use super::schema::{Edge, EdgeInput, EdgeType, Node, NodeInput, NodeLabel};

pub struct GraphStore {
    conn: Connection,
}

impl GraphStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating db directory {}", parent.display()))?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("opening SQLite db at {}", db_path.display()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA foreign_keys=ON;

             CREATE TABLE IF NOT EXISTS projects (
                 name        TEXT PRIMARY KEY,
                 repo_path   TEXT NOT NULL,
                 status      TEXT NOT NULL DEFAULT 'pending',
                 indexed_at  TEXT,
                 node_count  INTEGER NOT NULL DEFAULT 0,
                 edge_count  INTEGER NOT NULL DEFAULT 0
             );

             CREATE TABLE IF NOT EXISTS nodes (
                 id             INTEGER PRIMARY KEY AUTOINCREMENT,
                 project        TEXT NOT NULL,
                 label          TEXT NOT NULL,
                 qualified_name TEXT NOT NULL,
                 file_path      TEXT,
                 line_start     INTEGER,
                 line_end       INTEGER,
                 properties     TEXT NOT NULL DEFAULT '{}'
             );

             CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_project_qname
                 ON nodes(project, qualified_name);
             CREATE INDEX IF NOT EXISTS idx_nodes_project_label
                 ON nodes(project, label);
             CREATE INDEX IF NOT EXISTS idx_nodes_file
                 ON nodes(file_path);

             CREATE TABLE IF NOT EXISTS edges (
                 source_id  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                 target_id  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                 type       TEXT NOT NULL,
                 properties TEXT NOT NULL DEFAULT '{}',
                 PRIMARY KEY (source_id, target_id, type)
             );

             CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
             CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
             CREATE INDEX IF NOT EXISTS idx_edges_type   ON edges(type);

             CREATE TABLE IF NOT EXISTS embeddings (
                 node_id INTEGER PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
                 vector  BLOB NOT NULL
             );",
        )?;
        Ok(())
    }

    // ── Project ───────────────────────────────────────────────────────────────

    pub fn upsert_project(&self, name: &str, repo_path: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO projects(name, repo_path, status) VALUES(?1, ?2, 'indexing')
             ON CONFLICT(name) DO UPDATE SET repo_path=excluded.repo_path, status='indexing'",
            params![name, repo_path],
        )?;
        Ok(())
    }

    pub fn set_project_done(&self, name: &str) -> Result<()> {
        let node_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM nodes WHERE project=?1",
            params![name],
            |r| r.get(0),
        )?;
        let edge_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges e
              JOIN nodes n ON n.id=e.source_id
             WHERE n.project=?1",
            params![name],
            |r| r.get(0),
        )?;
        self.conn.execute(
            "UPDATE projects SET status='done', indexed_at=datetime('now'),
              node_count=?2, edge_count=?3
             WHERE name=?1",
            params![name, node_count, edge_count],
        )?;
        Ok(())
    }

    pub fn set_project_error(&self, name: &str, msg: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE projects SET status=?2 WHERE name=?1",
            params![name, format!("error: {}", msg)],
        )?;
        Ok(())
    }

    pub fn list_projects(&self) -> Result<Vec<serde_json::Value>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, repo_path, status, indexed_at, node_count, edge_count FROM projects ORDER BY name",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(serde_json::json!({
                "name":       r.get::<_, String>(0)?,
                "repo_path":  r.get::<_, String>(1)?,
                "status":     r.get::<_, String>(2)?,
                "indexed_at": r.get::<_, Option<String>>(3)?,
                "node_count": r.get::<_, i64>(4)?,
                "edge_count": r.get::<_, i64>(5)?,
            }))
        })?;
        rows.map(|r| r.map_err(anyhow::Error::from)).collect()
    }

    pub fn get_project_status(&self, name: &str) -> Result<Option<serde_json::Value>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, repo_path, status, indexed_at, node_count, edge_count
               FROM projects WHERE name=?1",
        )?;
        let mut rows = stmt.query_map(params![name], |r| {
            Ok(serde_json::json!({
                "name":       r.get::<_, String>(0)?,
                "repo_path":  r.get::<_, String>(1)?,
                "status":     r.get::<_, String>(2)?,
                "indexed_at": r.get::<_, Option<String>>(3)?,
                "node_count": r.get::<_, i64>(4)?,
                "edge_count": r.get::<_, i64>(5)?,
            }))
        })?;
        rows.next().transpose().map_err(anyhow::Error::from)
    }

    pub fn delete_project(&self, name: &str) -> Result<usize> {
        // Cascade deletes handle edges and embeddings
        let n = self
            .conn
            .execute("DELETE FROM nodes WHERE project=?1", params![name])?;
        self.conn
            .execute("DELETE FROM projects WHERE name=?1", params![name])?;
        Ok(n)
    }

    // ── Nodes ─────────────────────────────────────────────────────────────────

    /// Insert or update a node. Returns the node id.
    pub fn upsert_node(&self, node: &NodeInput) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO nodes(project, label, qualified_name, file_path, line_start, line_end, properties)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(project, qualified_name) DO UPDATE SET
               label=excluded.label,
               file_path=excluded.file_path,
               line_start=excluded.line_start,
               line_end=excluded.line_end,
               properties=excluded.properties",
            params![
                node.project,
                node.label.as_str(),
                node.qualified_name,
                node.file_path,
                node.line_start,
                node.line_end,
                node.properties.to_string(),
            ],
        )?;
        let id = self.conn.query_row(
            "SELECT id FROM nodes WHERE project=?1 AND qualified_name=?2",
            params![node.project, node.qualified_name],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    pub fn get_node_id(&self, project: &str, qualified_name: &str) -> Result<Option<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM nodes WHERE project=?1 AND qualified_name=?2",
        )?;
        let mut rows = stmt.query_map(params![project, qualified_name], |r| r.get(0))?;
        Ok(rows.next().transpose()?)
    }

    pub fn get_node_by_id(&self, id: i64) -> Result<Option<Node>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project, label, qualified_name, file_path, line_start, line_end, properties
               FROM nodes WHERE id=?1",
        )?;
        let mut rows = stmt.query_map(params![id], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i32>>(5)?,
                r.get::<_, Option<i32>>(6)?,
                r.get::<_, String>(7)?,
            ))
        })?;
        if let Some(row) = rows.next() {
            let (id, project, label_s, qname, file_path, ls, le, props_s) = row?;
            let label = NodeLabel::from_str(&label_s)
                .ok_or_else(|| anyhow::anyhow!("unknown label: {}", label_s))?;
            let properties = serde_json::from_str(&props_s).unwrap_or(serde_json::Value::Null);
            Ok(Some(Node {
                id,
                project,
                label,
                qualified_name: qname,
                file_path,
                line_start: ls,
                line_end: le,
                properties,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn search_nodes(
        &self,
        project: &str,
        pattern: &str,
        label: Option<NodeLabel>,
        limit: usize,
    ) -> Result<Vec<Node>> {
        let re = regex::Regex::new(pattern)
            .with_context(|| format!("invalid regex: {}", pattern))?;

        let label_filter = label.map(|l| format!("AND label='{}'", l.as_str())).unwrap_or_default();
        // No LIMIT in SQL — fetch all matching label rows and filter by regex in Rust.
        // This is correct for any pattern; for large tables (100K+ nodes) consider an FTS index.
        let sql = format!(
            "SELECT id, project, label, qualified_name, file_path, line_start, line_end, properties
               FROM nodes WHERE project=?1 {} ORDER BY qualified_name",
            label_filter
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![project], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i32>>(5)?,
                r.get::<_, Option<i32>>(6)?,
                r.get::<_, String>(7)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (id, project, label_s, qname, file_path, ls, le, props_s) = row?;
            if !re.is_match(&qname) {
                continue;
            }
            let label = NodeLabel::from_str(&label_s)
                .ok_or_else(|| anyhow::anyhow!("unknown label: {}", label_s))?;
            let properties = serde_json::from_str(&props_s).unwrap_or(serde_json::Value::Null);
            results.push(Node {
                id,
                project,
                label,
                qualified_name: qname,
                file_path,
                line_start: ls,
                line_end: le,
                properties,
            });
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    pub fn get_all_nodes(&self, project: &str) -> Result<Vec<Node>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project, label, qualified_name, file_path, line_start, line_end, properties
               FROM nodes WHERE project=?1",
        )?;
        let rows = stmt.query_map(params![project], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i32>>(5)?,
                r.get::<_, Option<i32>>(6)?,
                r.get::<_, String>(7)?,
            ))
        })?;
        let mut results = Vec::new();
        for row in rows {
            let (id, project, label_s, qname, file_path, ls, le, props_s) = row?;
            if let Some(label) = NodeLabel::from_str(&label_s) {
                let properties = serde_json::from_str(&props_s).unwrap_or(serde_json::Value::Null);
                results.push(Node {
                    id,
                    project,
                    label,
                    qualified_name: qname,
                    file_path,
                    line_start: ls,
                    line_end: le,
                    properties,
                });
            }
        }
        Ok(results)
    }

    // ── Edges ─────────────────────────────────────────────────────────────────

    pub fn upsert_edge(&self, edge: &EdgeInput) -> Result<()> {
        self.conn.execute(
            "INSERT INTO edges(source_id, target_id, type, properties)
             VALUES(?1, ?2, ?3, ?4)
             ON CONFLICT(source_id, target_id, type) DO UPDATE SET properties=excluded.properties",
            params![
                edge.source_id,
                edge.target_id,
                edge.edge_type.as_str(),
                edge.properties.to_string(),
            ],
        )?;
        Ok(())
    }

    pub fn get_neighbors(
        &self,
        node_id: i64,
        edge_type: Option<EdgeType>,
        direction: NeighborDirection,
    ) -> Result<Vec<(i64, EdgeType, serde_json::Value)>> {
        let type_filter = edge_type
            .map(|t| format!("AND e.type='{}'", t.as_str()))
            .unwrap_or_default();

        let sql = match direction {
            NeighborDirection::Outbound => format!(
                "SELECT e.target_id, e.type, e.properties FROM edges e WHERE e.source_id=?1 {}",
                type_filter
            ),
            NeighborDirection::Inbound => format!(
                "SELECT e.source_id, e.type, e.properties FROM edges e WHERE e.target_id=?1 {}",
                type_filter
            ),
            NeighborDirection::Both => format!(
                "SELECT e.target_id, e.type, e.properties FROM edges e WHERE e.source_id=?1 {}
                 UNION
                 SELECT e.source_id, e.type, e.properties FROM edges e WHERE e.target_id=?1 {}",
                type_filter, type_filter
            ),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![node_id], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (neighbor_id, edge_type_s, props_s) = row?;
            if let Some(et) = EdgeType::from_str(&edge_type_s) {
                let props = serde_json::from_str(&props_s).unwrap_or(serde_json::Value::Null);
                results.push((neighbor_id, et, props));
            }
        }
        Ok(results)
    }

    pub fn get_all_edges(&self, project: &str) -> Result<Vec<Edge>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.source_id, e.target_id, e.type, e.properties
               FROM edges e
               JOIN nodes n ON n.id = e.source_id
              WHERE n.project=?1",
        )?;
        let rows = stmt.query_map(params![project], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        let mut results = Vec::new();
        for row in rows {
            let (src, tgt, et_s, props_s) = row?;
            if let Some(et) = EdgeType::from_str(&et_s) {
                let props = serde_json::from_str(&props_s).unwrap_or(serde_json::Value::Null);
                results.push(Edge {
                    source_id: src,
                    target_id: tgt,
                    edge_type: et,
                    properties: props,
                });
            }
        }
        Ok(results)
    }

    // ── Embeddings ────────────────────────────────────────────────────────────

    pub fn upsert_embedding(&self, node_id: i64, vector: &[f32]) -> Result<()> {
        let blob: Vec<u8> = vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.conn.execute(
            "INSERT INTO embeddings(node_id, vector) VALUES(?1, ?2)
             ON CONFLICT(node_id) DO UPDATE SET vector=excluded.vector",
            params![node_id, blob],
        )?;
        Ok(())
    }

    pub fn get_embedding(&self, node_id: i64) -> Result<Option<Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector FROM embeddings WHERE node_id=?1")?;
        let mut rows = stmt.query_map(params![node_id], |r| r.get::<_, Vec<u8>>(0))?;
        if let Some(blob) = rows.next().transpose()? {
            let vec = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            Ok(Some(vec))
        } else {
            Ok(None)
        }
    }

    pub fn get_all_embeddings(&self, project: &str) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.node_id, e.vector FROM embeddings e
               JOIN nodes n ON n.id=e.node_id
              WHERE n.project=?1",
        )?;
        let rows = stmt.query_map(params![project], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            let (id, blob) = row?;
            let vec = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            results.push((id, vec));
        }
        Ok(results)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NeighborDirection {
    Inbound,
    Outbound,
    Both,
}

impl NeighborDirection {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "inbound" => Some(Self::Inbound),
            "outbound" => Some(Self::Outbound),
            "both" => Some(Self::Both),
            _ => None,
        }
    }
}
