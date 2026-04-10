# arborist-mcp — Specification

A C++ code intelligence MCP server in Rust. Single binary, no runtime dependencies beyond an optional system libclang.

---

## Overview

`arborist-mcp` exposes a persistent knowledge graph of C++ repositories to AI agents via the Model Context Protocol (JSON-RPC 2.0 over stdio). It:

1. Parses C++ source into a SQLite knowledge graph
2. Computes Cleora-style node embeddings over the graph
3. Exposes 10 MCP tools for querying, traversal, and semantic search

---

## Architecture

```
arborist-mcp/
  src/
    main.rs              — entry point, stdio MCP loop
    mcp/
      mod.rs             — tool registration + dispatch
      tools.rs           — tool handler implementations
    parser/
      mod.rs             — file discovery, ingestion, call/include resolution
      treesitter.rs      — tree-sitter extraction (primary parser)
      libclang.rs        — libclang resolution (supplementary, optional)
    graph/
      mod.rs             — GraphBuffer (write-batching layer)
      store.rs           — SQLite read/write (GraphStore)
      schema.rs          — NodeLabel, EdgeType, Node, Edge types
    embeddings/
      mod.rs             — compute_and_store, find_similar
      features.rs        — initial feature vector construction
      propagation.rs     — iterative Cleora propagation + L2 normalization
    pipeline/
      mod.rs             — multi-pass indexing orchestration (rayon)
  Cargo.toml
```

---

## Technology Stack

| Component | Crate | Notes |
|-----------|-------|-------|
| MCP protocol | `rmcp` (github) | `server` + `transport-io` features |
| Primary parser | `tree-sitter` + `tree-sitter-cpp` 0.23 | Structural extraction, no build system required |
| Supplementary parser | `clang-sys` 1.x | Runtime probe; activated only when `compile_commands.json` present |
| Graph storage | `rusqlite` 0.31 (bundled) | Statically linked libsqlite3 |
| Embeddings | `ndarray` 0.16 | Pure Rust Cleora implementation |
| Parallelism | `rayon` | Parallel file parsing |
| Async runtime | `tokio` | For MCP server loop |
| Hashing | `fnv` | FNV-1a for feature bucketing |

---

## Graph Schema

### SQLite tables

```sql
projects (
    name        TEXT PRIMARY KEY,
    repo_path   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- 'indexing' | 'done' | 'error: ...'
    indexed_at  TEXT,
    node_count  INTEGER NOT NULL DEFAULT 0,
    edge_count  INTEGER NOT NULL DEFAULT 0
);

nodes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project        TEXT NOT NULL,
    label          TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    file_path      TEXT,
    line_start     INTEGER,
    line_end       INTEGER,
    properties     TEXT NOT NULL DEFAULT '{}'
);
UNIQUE INDEX on (project, qualified_name)
INDEX on (project, label)
INDEX on (file_path)

edges (
    source_id  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type       TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, type)
);
INDEX on source_id, target_id, type

embeddings (
    node_id INTEGER PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    vector  BLOB NOT NULL   -- float32 array, little-endian
);
```

WAL mode and foreign keys are enabled on every connection.

### Node labels

| Label | Description |
|-------|-------------|
| `Project` | Root project node |
| `File` | Source/header file |
| `Namespace` | C++ namespace |
| `Class` | Class definition |
| `Struct` | Struct definition |
| `Function` | Free function |
| `Method` | Member function |
| `Template` | Template declaration |
| `Enum` | Enum definition |
| `Variable` | Variable / field |

### Edge types and propagation weights

| Edge | Weight | Description |
|------|--------|-------------|
| `CALLS` | 1.0 | Function calls another function |
| `INHERITS` | 1.2 | Class derives from base class |
| `OVERRIDES` | 1.1 | Method overrides virtual method |
| `INSTANTIATES` | 0.9 | Template instantiation |
| `USES_TYPE` | 0.8 | Symbol uses a type |
| `DEFINES` | 0.7 | File defines a symbol |
| `INCLUDES` | 0.6 | File includes another file |
| `CONTAINS` | 0.5 | Scope contains a symbol |

---

## Parsing Pipeline

Seven sequential passes per indexing run:

| Pass | What happens |
|------|-------------|
| 1 | Discover all C++ files (rayon parallel) and parse with tree-sitter |
| 2 | Upsert a `Project` root node |
| 3 | Ingest symbols: create `File` nodes and symbol nodes; add `DEFINES`/`CONTAINS` edges |
| 4 | Wire `CALLS` edges via fuzzy regex matching of call sites |
| 5 | Wire `INCLUDES` edges from `#include` directives |
| 6 | Optional libclang pass: type-resolved `CALLS`, `INHERITS`, `USES_TYPE` edges (skipped if no `compile_commands.json` or no libclang) |
| 7 | Compute Cleora embeddings and store in `embeddings` table |

### Tree-sitter extraction (primary)

Extracts from AST for each file:
- Symbols: `namespace_definition`, `class_specifier`, `struct_specifier`, `function_definition`, `template_declaration`, `enum_specifier`
- Qualified names built by tracking parent scopes during recursive walk
- Call sites collected inside function bodies via `call_expression` nodes

File collection skips common build/vendor directories and matches extensions: `.cpp`, `.cc`, `.cxx`, `.c`, `.h`, `.hpp`, `.hxx`, `.h++`.

### libclang supplementary resolution

Activated when:
1. `compile_commands.json` exists (at provided path or `<repo_root>/compile_commands.json`)
2. A system libclang shared library is found at a platform-specific probe path

Provides:
- Type-resolved `CALLS` edges (handles overloading, templates)
- `INHERITS` edges from `CXXBaseSpecifier` cursors
- `USES_TYPE` edges from `TypeRef` cursors

Gracefully degrades to tree-sitter-only if either condition is not met.

---

## Cleora Embeddings

### Feature vector layout

Dimension: **90** total (`HASH_DIMS=64` + `LABEL_DIMS=10` + `DEGREE_DIMS=16`)

```
[0..64)    FNV-1a hash of qualified_name tokens (split on "::")
[64..74)   Label one-hot (10 node types)
[74..90)   Log-normalized in/out degree per edge type (8 types × 2 directions)
```

Log normalization: `ln(1 + degree_count)`.

All initial feature vectors are L2-normalized before propagation.

### Propagation algorithm

```
for iter in 0..4:
    for each node v:
        agg = weighted_mean(neighbor_embeddings, weights=edge_weights)
        combined = v * 0.5 + agg * 0.5
        combined = L2_normalize(combined)
```

- 4 iterations (`NUM_ITERS`)
- Self-loop weight `alpha = 0.5`
- Undirected adjacency (edges added in both directions)
- Parallel edges between same pair of nodes have their weights summed
- Isolated nodes (no neighbors) retain their initial embedding

Similarity: cosine similarity (dot product of unit vectors).

---

## MCP Tools

| Tool | Required params | Optional params | Returns |
|------|----------------|-----------------|---------|
| `index_repository` | `repo_path` | `compile_commands_path` | `{ status, project, message }` — starts background thread |
| `list_projects` | — | — | `{ projects: [...] }` with node/edge counts |
| `index_status` | `project` | — | Project record with `status`, `node_count`, `edge_count`, `indexed_at` |
| `search_graph` | `project`, `name_pattern` | `label`, `limit` (default 20) | `{ results: [...], count }` |
| `trace_calls` | `project`, `function_name` | `direction` (default `outbound`), `depth` (default 3, max 10) | `{ root, trace: [...] }` BFS over CALLS edges |
| `find_similar` | `project`, `function_name` | `top_k` (default 10) | `{ anchor, similar: [{ similarity, ... }] }` |
| `get_snippet` | `project`, `qualified_name` | — | `{ snippet, file_path, line_start, line_end }` — reads live source file |
| `detect_changes` | `project` | — | `{ changed_files, affected_symbols, count }` — runs `git diff --name-only HEAD` |
| `query` | `project`, `query` | — | `{ nodes, count }` and optionally `edges` for relationship patterns |
| `delete_project` | `project` | — | `{ deleted, nodes_removed }` — removes DB file |

### `query` tool syntax

Minimal subset of Cypher-like syntax parsed with regex:

```
MATCH (n:Label {name: "pattern"}) RETURN n LIMIT k
MATCH (n)-[r:TYPE]->(m) WHERE n.name = "pattern" RETURN n, r, m LIMIT k
```

Label and name pattern are optional. Relationship patterns (`->`, `<-`, `-[`) trigger edge inclusion in response.

### `trace_calls` directions

- `outbound` — who does this function call?
- `inbound` — who calls this function?
- `both` — full neighborhood

### `index_repository` behavior

- Project name is derived from the last path component of `repo_path`
- Indexing runs in a background thread; the tool response returns immediately
- Poll with `index_status` to check progress
- Status transitions: `indexing` → `done` or `error: <message>`

---

## Storage

Each project gets its own SQLite file: `<db_dir>/<project_name>.db`

Project names are sanitized: only `[a-zA-Z0-9_-]` are kept; other characters become `_`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARBORIST_CACHE_DIR` | `~/.cache/arborist-mcp` (macOS: `~/Library/Caches/arborist-mcp`) | SQLite storage directory |
| `ARBORIST_LOG` | `arborist_mcp=info` | tracing log filter (stderr only) |

---

## Build

```toml
[dependencies]
tree-sitter = "0.23"
tree-sitter-cpp = "0.23"
clang-sys = { version = "1", features = ["runtime", "clang_14_0"] }
rusqlite = { version = "0.31", features = ["bundled"] }
ndarray = "0.16"
rmcp = { git = "https://github.com/modelcontextprotocol/rust-sdk", package = "rmcp", features = ["server", "transport-io"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
rayon = "1"
regex = "1"
walkdir = "2"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
dirs = "5"
fnv = "1"
```

```bash
# Development
cargo build

# Release (macOS — dynamic links only system frameworks)
cargo build --release

# Static binary (Linux)
cargo build --release --target x86_64-unknown-linux-musl
```

---

## What is NOT supported

- Languages other than C++
- Graph visualization UI
- Install / uninstall commands
- Multi-language dispatch
- Telemetry or update checks
- Claude skill files or agent auto-detection

---

## Definition of Done

- Binary starts and responds correctly to MCP `initialize`
- `index_repository` on a real C++ project produces nodes + edges in SQLite
- Cleora embeddings computed and stored after indexing
- `find_similar` returns plausible results for a known function
- `search_graph` and `trace_calls` return correct results
- Binary is self-contained (no runtime deps beyond optional system libclang)
