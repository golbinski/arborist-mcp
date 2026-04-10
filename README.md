# arborist-mcp

A C++ code intelligence MCP server in Rust. Parses C++ repositories into a persistent SQLite knowledge graph, computes Cleora-style node embeddings, and exposes 10 MCP tools to AI agents over stdio.

## Features

- **Two-layer parsing**: tree-sitter (fast structural extraction) + libclang (type-resolved calls, template instantiation, overloads — when `compile_commands.json` is present)
- **Persistent knowledge graph**: SQLite with nodes (classes, functions, namespaces, ...) and typed edges (CALLS, INHERITS, INCLUDES, ...)
- **Semantic embeddings**: Cleora algorithm — iterative feature propagation on the unit hypersphere, pure Rust implementation
- **10 MCP tools**: search, BFS traversal, semantic similarity, source snippets, change detection
- **Self-contained binary**: statically linked SQLite, runtime-optional libclang
- **VCS-aware file discovery**: respects `.gitignore`, `.hgignore`, `.ignore`

## Installation

```bash
git clone https://github.com/golbinski/arborist-mcp
cd arborist-mcp
cargo build --release
```

The binary is at `target/release/arborist-mcp`. Copy it anywhere on your `PATH`.

For a fully static binary on Linux:

```bash
cargo build --release --target x86_64-unknown-linux-musl
```

### libclang (optional)

Type-resolved analysis (overloads, templates, inheritance) requires libclang. The server auto-discovers it by running `clang -print-search-dirs` and `llvm-config --libdir`. If `clang` is in your `PATH`, no manual configuration is needed:

```bash
# macOS
xcode-select --install        # CommandLineTools ships libclang

# Ubuntu/Debian
apt install clang              # ships libclang.so
```

If libclang is not found, the server falls back to tree-sitter-only extraction — all tools still work, call edges are less precise.

## MCP Configuration

Add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "arborist": {
      "command": "/path/to/arborist-mcp",
      "env": {
        "ARBORIST_LOG": "arborist_mcp=info"
      }
    }
  }
}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARBORIST_CACHE_DIR` | `~/.cache/arborist-mcp` (macOS: `~/Library/Caches/arborist-mcp`) | Directory for SQLite databases |
| `ARBORIST_LOG` | `arborist_mcp=info` | Log level filter (output goes to stderr, not stdio) |

## Usage

### Index a repository

```
index_repository(repo_path="/path/to/myrepo")
```

Returns immediately; indexing runs in the background. Poll with `index_status`.

With a compilation database for type-resolved analysis:

```
index_repository(
  repo_path="/path/to/myrepo",
  compile_commands_path="/path/to/myrepo/build/compile_commands.json"
)
```

### Check indexing status

```
index_status(project="myrepo")
```

Returns: `{ "status": "done", "node_count": 24324, "edge_count": 88973, "indexed_at": "..." }`

Status values: `indexing` | `done` | `error: <message>`

### Search for symbols

```
search_graph(project="myrepo", name_pattern="IOBuf", label="Class", limit=10)
```

`name_pattern` is a regex matched against `qualified_name`. `label` is optional and filters to one node type.

Node labels: `Project` `File` `Namespace` `Class` `Struct` `Function` `Method` `Template` `Enum` `Variable`

### Trace call graph

```
trace_calls(project="myrepo", function_name="folly::IOBuf::copyBuffer", direction="outbound", depth=3)
```

BFS over `CALLS` edges. `direction`: `outbound` (who does it call?), `inbound` (who calls it?), `both`.

### Find semantically similar symbols

```
find_similar(project="myrepo", function_name="folly::Future", top_k=10)
```

Returns symbols ranked by Cleora embedding cosine similarity. Tends to surface structurally similar symbols — classes with similar neighbour patterns, functions with similar call profiles.

### Get source snippet

```
get_snippet(project="myrepo", qualified_name="folly::IOBuf::copyBuffer")
```

Returns the source lines for the symbol from the live file on disk.

### Detect changes

```
detect_changes(project="myrepo")
```

Runs `git diff --name-only HEAD` in the repo directory and maps changed files to the symbols defined in them.

### Graph query

```
query(project="myrepo", query="MATCH (n:Function {name: \"parse\"}) RETURN n LIMIT 20")
query(project="myrepo", query="MATCH (n)-[r:CALLS]->(m) WHERE n.name = \"main\" RETURN n, r, m")
```

Supports a minimal Cypher-like syntax: `MATCH`, `WHERE`, `RETURN`, `LIMIT`. Relationship patterns (`->`, `<-`, `-[`) include edges in the response.

### List projects

```
list_projects()
```

### Delete a project

```
delete_project(project="myrepo")
```

Removes all graph data and the database file.

## Knowledge graph

### Node labels

| Label | Description |
|-------|-------------|
| `Project` | Root node for the indexed repository |
| `File` | Source or header file |
| `Namespace` | C++ namespace |
| `Class` | Class definition |
| `Struct` | Struct definition |
| `Function` | Free function |
| `Method` | Member function |
| `Template` | Template declaration |
| `Enum` | Enum definition |
| `Variable` | Variable or field |

### Edge types

| Edge | Description |
|------|-------------|
| `CONTAINS` | Scope contains a symbol |
| `DEFINES` | File defines a symbol |
| `CALLS` | Function calls another function |
| `INCLUDES` | File includes another file |
| `INHERITS` | Class derives from base class |
| `INSTANTIATES` | Template instantiation |
| `OVERRIDES` | Method overrides virtual method |
| `USES_TYPE` | Symbol uses a type |

### Storage

Each project is stored in its own SQLite file at `<ARBORIST_CACHE_DIR>/<project_name>.db`. Project names are derived from the last component of `repo_path` and sanitized to `[a-zA-Z0-9_-]`.

## Architecture

```
src/
  main.rs              — stdio MCP server entry point
  mcp/
    mod.rs             — tool registration and dispatch
    tools.rs           — tool handler implementations
  parser/
    mod.rs             — file discovery (.gitignore-aware), ingestion passes
    treesitter.rs      — tree-sitter AST extraction (primary)
    libclang.rs        — libclang resolution (supplementary, optional)
  graph/
    mod.rs             — GraphBuffer (write-batching layer)
    store.rs           — SQLite read/write, in-memory + backup API
    schema.rs          — NodeLabel, EdgeType, Node, Edge types
  embeddings/
    mod.rs             — compute_and_store, find_similar
    features.rs        — initial feature vector construction
    propagation.rs     — Cleora iterative propagation + L2 normalization
  pipeline/
    mod.rs             — 7-pass indexing pipeline (rayon-parallel)
```

### Indexing pipeline

All writes go into an in-memory SQLite database during indexing (no fsync overhead). When indexing completes the database is persisted to disk in a single bulk backup operation.

Seven passes:

1. Discover C++ files respecting `.gitignore` / `.hgignore` / `.ignore` — parallel parse with tree-sitter
2. Create `Project` root node
3. Ingest symbols: `File` nodes, symbol nodes, `DEFINES` / `CONTAINS` edges
4. Wire `CALLS` edges — O(nodes + calls) in-memory index lookup
5. Wire `INCLUDES` edges
6. Optional libclang pass: type-resolved `CALLS`, `INHERITS`, `USES_TYPE` edges
7. Compute and store Cleora embeddings

### Embeddings

Feature vector: 90 dimensions — 64 FNV-1a name hash + 10 label one-hot + 16 log-normalized degree features (in/out per edge type).

Cleora propagation: 4 iterations, α=0.5 self-loop weight, undirected adjacency, edge-type-weighted mean pooling, L2-normalization after each iteration.

## Development

```bash
cargo build          # debug build
cargo test           # run unit tests (19 tests)
cargo build --release
```

Tests cover: schema round-trips, embedding dimensions and unit-length invariant, propagation stability, cosine similarity, file discovery with ignore rules.
