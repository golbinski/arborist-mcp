mod embeddings;
mod graph;
mod mcp;
mod parser;
mod pipeline;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rmcp::{transport::stdio, ServiceExt};
use serde_json::{json, Value};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use mcp::tools::ToolContext;
use mcp::ArboristServer;

// ── CLI definition ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "arborist-mcp",
    about = "C++ code intelligence tool.\nRun without arguments to start the MCP server over stdio."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Index a C++ repository (blocks until complete)
    Index {
        /// Path to the repository root
        repo_path: String,
        /// Optional path to compile_commands.json for type-resolved analysis
        #[arg(long)]
        compile_commands: Option<String>,
    },
    /// Check the indexing status of a project
    Status {
        /// Project name
        project: String,
    },
    /// List all indexed projects
    List,
    /// Search for symbols by name pattern (regex)
    Search {
        /// Project name
        project: String,
        /// Regex pattern matched against qualified_name
        pattern: String,
        /// Restrict to one node type
        #[arg(long)]
        label: Option<String>,
        /// Maximum results
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// BFS traversal over CALLS edges from a function
    Trace {
        /// Project name
        project: String,
        /// Qualified function name (regex prefix match)
        function: String,
        /// Edge direction: outbound | inbound | both
        #[arg(long, default_value = "outbound")]
        direction: String,
        /// BFS hop depth (max 10)
        #[arg(long, default_value = "3")]
        depth: usize,
    },
    /// Find semantically similar symbols by Cleora embedding cosine similarity
    Similar {
        /// Project name
        project: String,
        /// Qualified symbol name
        symbol: String,
        /// Number of results
        #[arg(long, default_value = "10")]
        top: usize,
    },
    /// Return the source lines for a symbol
    Snippet {
        /// Project name
        project: String,
        /// Qualified name of the symbol
        qualified_name: String,
    },
    /// Map `git diff HEAD` changed files to affected symbols
    Changes {
        /// Project name
        project: String,
    },
    /// Run a graph query (minimal Cypher-like syntax)
    Query {
        /// Project name
        project: String,
        /// Query string, e.g. 'MATCH (n:Function {name: "parse"}) RETURN n LIMIT 20'
        query: String,
    },
    /// Generate a standalone interactive HTML graph visualization
    Export {
        /// Project name
        project: String,
        /// Output path for the .html file
        output: String,
        /// Qualified name of the node to BFS from (omit for high-degree overview)
        #[arg(long)]
        root: Option<String>,
        /// Restrict subgraph to one node type
        #[arg(long)]
        label: Option<String>,
        /// BFS hop depth
        #[arg(long, default_value = "3")]
        depth: usize,
        /// Maximum nodes to include
        #[arg(long, default_value = "200")]
        max_nodes: usize,
    },
    /// Remove a project and all its graph data
    Delete {
        /// Project name
        project: String,
    },
    /// Start MCP server over stdio (default when no subcommand)
    Serve,
}

// ── Entry point ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        None | Some(Command::Serve) => run_mcp_server().await,
        Some(cmd) => {
            // CLI mode: log to stderr at warn level by default so info noise
            // doesn't mix with JSON output on stdout.
            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::from_env("ARBORIST_LOG").add_directive("arborist_mcp=warn".parse()?),
                )
                .with_writer(std::io::stderr)
                .init();

            let db_dir = db_directory();
            std::fs::create_dir_all(&db_dir)?;
            run_cli(cmd, db_dir)
        }
    }
}

async fn run_mcp_server() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_env("ARBORIST_LOG").add_directive("arborist_mcp=info".parse()?),
        )
        .with_writer(std::io::stderr)
        .init();

    let db_dir = db_directory();
    std::fs::create_dir_all(&db_dir)?;
    tracing::info!("arborist-mcp starting, db_dir={}", db_dir.display());

    let server = ArboristServer::new(db_dir);
    server
        .serve(stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP serve error: {}", e))?
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP wait error: {}", e))?;
    Ok(())
}

fn run_cli(cmd: Command, db_dir: PathBuf) -> Result<()> {
    let ctx = ToolContext { db_dir };

    match cmd {
        Command::Index {
            repo_path,
            compile_commands,
        } => {
            // Run pipeline directly (blocking) so the user sees a clean exit.
            let project_name = std::path::Path::new(&repo_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unnamed".to_owned());

            eprintln!("Indexing {} → project \"{}\" …", repo_path, project_name);

            let db_path = ctx.db_dir.join(format!("{}.db", sanitize(&project_name)));
            let config = pipeline::IndexingConfig {
                repo_path: PathBuf::from(&repo_path),
                project_name: project_name.clone(),
                compile_commands_path: compile_commands.map(PathBuf::from),
                db_path,
            };
            pipeline::run(config)?;

            // Read back stats
            let result = mcp::tools::index_status(&ctx, &json!({"project": project_name}))?;
            print_result(&result);
        }

        Command::Status { project } => {
            let result = mcp::tools::index_status(&ctx, &json!({"project": project}))?;
            print_result(&result);
        }

        Command::List => {
            let result = mcp::tools::list_projects(&ctx, &json!({}))?;
            print_result(&result);
        }

        Command::Search {
            project,
            pattern,
            label,
            limit,
        } => {
            let mut p = json!({"project": project, "name_pattern": pattern, "limit": limit});
            if let Some(l) = label {
                p["label"] = Value::String(l);
            }
            let result = mcp::tools::search_graph(&ctx, &p)?;
            print_result(&result);
        }

        Command::Trace {
            project,
            function,
            direction,
            depth,
        } => {
            let result = mcp::tools::trace_calls(
                &ctx,
                &json!({
                    "project": project,
                    "function_name": function,
                    "direction": direction,
                    "depth": depth
                }),
            )?;
            print_result(&result);
        }

        Command::Similar {
            project,
            symbol,
            top,
        } => {
            let result = mcp::tools::find_similar(
                &ctx,
                &json!({
                    "project": project,
                    "function_name": symbol,
                    "top_k": top
                }),
            )?;
            print_result(&result);
        }

        Command::Snippet {
            project,
            qualified_name,
        } => {
            let result = mcp::tools::get_snippet(
                &ctx,
                &json!({
                    "project": project,
                    "qualified_name": qualified_name
                }),
            )?;
            print_result(&result);
        }

        Command::Changes { project } => {
            let result = mcp::tools::detect_changes(&ctx, &json!({"project": project}))?;
            print_result(&result);
        }

        Command::Query { project, query } => {
            let result =
                mcp::tools::query_graph(&ctx, &json!({"project": project, "query": query}))?;
            print_result(&result);
        }

        Command::Export {
            project,
            output,
            root,
            label,
            depth,
            max_nodes,
        } => {
            let mut p = json!({
                "project": project,
                "output_path": output,
                "depth": depth,
                "max_nodes": max_nodes
            });
            if let Some(r) = root {
                p["root_node"] = Value::String(r);
            }
            if let Some(l) = label {
                p["label"] = Value::String(l);
            }
            let result = mcp::tools::export_graph(&ctx, &p)?;
            print_result(&result);
        }

        Command::Delete { project } => {
            let result = mcp::tools::delete_project(&ctx, &json!({"project": project}))?;
            print_result(&result);
        }

        Command::Serve => unreachable!(),
    }

    Ok(())
}

fn print_result(v: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
    );
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn db_directory() -> PathBuf {
    if let Ok(dir) = std::env::var("ARBORIST_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("arborist-mcp")
}
