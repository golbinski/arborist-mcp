mod embeddings;
mod graph;
mod mcp;
mod parser;
mod pipeline;

use anyhow::Result;
use rmcp::{ServiceExt, transport::stdio};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use mcp::ArboristServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Logging to stderr so it doesn't corrupt the stdio JSON-RPC stream
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
    let transport = stdio();

    server
        .serve(transport)
        .await
        .map_err(|e| anyhow::anyhow!("MCP serve error: {}", e))?
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP wait error: {}", e))?;

    Ok(())
}

fn db_directory() -> PathBuf {
    if let Ok(dir) = std::env::var("ARBORIST_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("arborist-mcp")
}
