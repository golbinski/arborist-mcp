# arborist-mcp — Build Prompt

Build a C++ code intelligence MCP server in Rust. Call it `arborist-mcp`.

## Goal

A single-binary MCP server (JSON-RPC 2.0 over stdio) that:
1. Parses C++ source into a persistent knowledge graph (SQLite)
2. Computes Cleora-style node embeddings over the graph
3. Exposes the graph and embeddings via MCP tools to AI agents

No Claude skills, no agent auto-detection, no install scripts — just the binary and an MCP server.

## Reference implementation

Study `/Users/golbinski/Work/ext/codebase-memory-mcp` for:
- Graph schema design (nodes/edges tables, node labels, edge types)
- What gets extracted from C++ (`internal/cbm/extract_defs.c`, look for `CBM_LANG_CPP` branches)
- Pipeline pass structure (`src/pipeline/`)
- MCP tool definitions and JSON-RPC framing (`src/mcp/mcp.c`)
- The semantic signal design (`src/semantic/semantic.c`) — replace with Cleora embeddings

## Technology stack

**Parsing — two-layer approach:**

- **tree-sitter** (`tree-sitter` crate + `tree-sitter-cpp`) as primary parser:
  fast structural extraction — symbols, files, namespaces, basic call sites.
  Works without a build system, tolerates incomplete/broken code.
- **libclang** (`clang-sys` crate) as supplementary resolver:
  type-resolved calls, template instantiation, overload resolution.
  Activated only when `compile_commands.json` is present; gracefully degrade if absent.

**Graph storage:**
- SQLite via `rusqlite` with `bundled` feature (statically links libsqlite3 — no system dependency)
- Schema:
  ```sql
  nodes(id INTEGER PK, project TEXT, label TEXT, qualified_name TEXT,
        file_path TEXT, line_start INT, line_end INT, properties TEXT)
  edges(source_id INT, target_id INT, type TEXT, properties TEXT)
  embeddings(node_id INT PK, vector BLOB)  -- float32 array, little-endian
  ```
- Node labels: `Project`, `File`, `Namespace`, `Class`, `Struct`, `Function`, `Method`, `Template`, `Enum`, `Variable`
- Edge types: `CONTAINS`, `DEFINES`, `CALLS`, `INCLUDES`, `INHERITS`, `INSTANTIATES`, `OVERRIDES`, `USES_TYPE`

**Embeddings — Cleora algorithm:**

Cleora is iterative feature propagation on a hypersphere. Implement in pure Rust using `ndarray`:

1. Initialize each node with a feature vector:
   - Name token hash (FNV-1a, bucketed into N dimensions)
   - Label one-hot
   - Log-normalized in/out degree per edge type
2. For each iteration (3–5 sufficient):
   a. For each node: aggregate neighbor vectors (mean pooling, weighted by edge type)
   b. L2-normalize the result back onto the unit hypersphere
3. Store final `float32` embeddings as BLOB in `embeddings` table
4. Expose cosine similarity search via MCP tool

Reference: Synerise/cleora on GitHub for the original algorithm description. Implement from scratch in Rust using `ndarray` rather than binding to the Cleora binary — the algorithm is ~300 LOC and embedding it avoids a runtime dependency.

**MCP server:**
- Use `modelcontextprotocol/rust-sdk` (`rmcp` crate) for MCP protocol handling
- Transport: stdio (JSON-RPC 2.0 over stdin/stdout)
- If `rmcp` API is insufficient, implement minimal JSON-RPC 2.0 framing directly

## MCP tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `index_repository` | `repo_path`, `compile_commands_path?` | Parse + build graph + compute embeddings |
| `list_projects` | — | Show indexed projects with node/edge counts |
| `index_status` | `project` | Check indexing progress/completion |
| `search_graph` | `project`, `name_pattern`, `label?`, `limit?` | Regex search over `qualified_name` |
| `trace_calls` | `project`, `function_name`, `direction` (inbound/outbound/both), `depth` | BFS over CALLS edges |
| `find_similar` | `project`, `function_name`, `top_k` | Cosine similarity over Cleora embeddings |
| `get_snippet` | `project`, `qualified_name` | Return source lines for a symbol |
| `detect_changes` | `project` | Map `git diff` to affected symbols |
| `query` | `project`, `query` | Basic MATCH/WHERE/RETURN graph queries |
| `delete_project` | `project` | Remove project and all graph data |

## What NOT to build

- No Claude skill files or instruction files
- No agent auto-detection or MCP config injection
- No install/uninstall commands
- No graph visualization UI
- No language support beyond C++
- No update checker or telemetry
- No multi-language dispatch table

## Build

```toml
# Cargo.toml key dependencies
tree-sitter = "0.23"
tree-sitter-cpp = "0.23"
clang-sys = { version = "1", features = ["runtime", "clang_14_0"] }
rusqlite = { version = "0.31", features = ["bundled"] }
ndarray = "0.16"
rmcp = { git = "https://github.com/modelcontextprotocol/rust-sdk" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
rayon = "1"          # parallel file indexing
regex = "1"
walkdir = "2"
```

Static binary (Linux):
```bash
cargo build --release --target x86_64-unknown-linux-musl
```

macOS produces a portable binary without musl; dynamic links only system frameworks.

Config via environment:
- `ARBORIST_CACHE_DIR` — SQLite storage dir (default: `~/.cache/arborist-mcp`)
- `COMPILE_COMMANDS_PATH` — override compile_commands.json location

## Project structure

```
arborist-mcp/
  src/
    main.rs                 — entry point, MCP stdio loop
    mcp/
      mod.rs                — tool registration + dispatch
      tools.rs              — tool handler implementations
    parser/
      mod.rs                — orchestrates tree-sitter + libclang passes
      treesitter.rs         — tree-sitter extraction (primary)
      libclang.rs           — libclang resolution (supplementary)
    graph/
      mod.rs                — graph builder
      store.rs              — SQLite read/write layer
      schema.rs             — node/edge types, migrations
    embeddings/
      mod.rs                — Cleora implementation
      features.rs           — initial node feature construction
      propagation.rs        — iterative propagation + L2 normalization
    pipeline/
      mod.rs                — multi-pass indexing orchestration (rayon)
  Cargo.toml
```

## Definition of done

- `arborist-mcp` binary starts and responds correctly to MCP `initialize`
- `index_repository` on a real C++ project produces nodes + edges in SQLite
- Cleora embeddings computed and stored after indexing
- `find_similar` returns plausible results for a known function
- `search_graph` and `trace_calls` return correct results
- Binary is self-contained (no runtime deps beyond libclang shared lib)
