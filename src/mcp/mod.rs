pub mod tools;

use rmcp::{
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
    },
    service::RequestContext,
    ErrorData as McpError, RoleServer, ServerHandler,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

use tools::ToolContext;

#[derive(Clone)]
pub struct ArboristServer {
    ctx: Arc<ToolContext>,
}

impl ArboristServer {
    pub fn new(db_dir: PathBuf) -> Self {
        Self {
            ctx: Arc::new(ToolContext { db_dir }),
        }
    }
}

impl ServerHandler for ArboristServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("arborist-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "C++ code intelligence server. Index a repo with index_repository, \
                 then explore the knowledge graph with search_graph, trace_calls, find_similar, etc.",
            )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(all_tool_definitions()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let params: Value = request
            .arguments
            .map(Value::Object)
            .unwrap_or_else(|| Value::Object(Default::default()));
        let ctx = &self.ctx;

        let result: anyhow::Result<Value> = match request.name.as_ref() {
            "index_repository" => tools::index_repository(ctx, &params),
            "list_projects" => tools::list_projects(ctx, &params),
            "index_status" => tools::index_status(ctx, &params),
            "search_graph" => tools::search_graph(ctx, &params),
            "trace_calls" => tools::trace_calls(ctx, &params),
            "find_similar" => tools::find_similar(ctx, &params),
            "get_snippet" => tools::get_snippet(ctx, &params),
            "detect_changes" => tools::detect_changes(ctx, &params),
            "query" => tools::query_graph(ctx, &params),
            "delete_project" => tools::delete_project(ctx, &params),
            other => Err(anyhow::anyhow!("unknown tool: {}", other)),
        };

        match result {
            Ok(v) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string()),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Error: {}",
                e
            ))])),
        }
    }
}

fn make_tool(name: &'static str, description: &'static str, schema: Value) -> Tool {
    let schema_obj = schema.as_object().cloned().unwrap_or_default();
    Tool::new(name, description, Arc::new(schema_obj))
}

fn all_tool_definitions() -> Vec<Tool> {
    vec![
        make_tool(
            "index_repository",
            "Parse a C++ repository, build the knowledge graph, and compute Cleora embeddings.",
            json!({
                "type": "object",
                "properties": {
                    "repo_path": { "type": "string", "description": "Absolute path to the repository root" },
                    "compile_commands_path": { "type": "string", "description": "Optional path to compile_commands.json" }
                },
                "required": ["repo_path"]
            }),
        ),
        make_tool(
            "list_projects",
            "List all indexed projects with node and edge counts.",
            json!({ "type": "object", "properties": {} }),
        ),
        make_tool(
            "index_status",
            "Check the indexing status of a project.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string", "description": "Project name" }
                },
                "required": ["project"]
            }),
        ),
        make_tool(
            "search_graph",
            "Regex search over qualified names in the knowledge graph.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "name_pattern": { "type": "string", "description": "Regex pattern to match qualified_name" },
                    "label": {
                        "type": "string",
                        "enum": ["Project","File","Namespace","Class","Struct","Function","Method","Template","Enum","Variable"]
                    },
                    "limit": { "type": "integer", "default": 20 }
                },
                "required": ["project", "name_pattern"]
            }),
        ),
        make_tool(
            "trace_calls",
            "BFS traversal over CALLS edges from a function.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "function_name": { "type": "string" },
                    "direction": { "type": "string", "enum": ["inbound", "outbound", "both"], "default": "outbound" },
                    "depth": { "type": "integer", "default": 3, "maximum": 10 }
                },
                "required": ["project", "function_name"]
            }),
        ),
        make_tool(
            "find_similar",
            "Find functions/classes similar to a given symbol by Cleora embedding cosine similarity.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "function_name": { "type": "string" },
                    "top_k": { "type": "integer", "default": 10 }
                },
                "required": ["project", "function_name"]
            }),
        ),
        make_tool(
            "get_snippet",
            "Return the source lines for a symbol given its qualified name.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "qualified_name": { "type": "string" }
                },
                "required": ["project", "qualified_name"]
            }),
        ),
        make_tool(
            "detect_changes",
            "Map `git diff HEAD` changed files to affected symbols in the graph.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" }
                },
                "required": ["project"]
            }),
        ),
        make_tool(
            "query",
            "Basic graph query. Syntax: MATCH (n:Label {name: \"pattern\"}) RETURN n LIMIT k",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "query": { "type": "string" }
                },
                "required": ["project", "query"]
            }),
        ),
        make_tool(
            "delete_project",
            "Remove a project and all its graph data.",
            json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" }
                },
                "required": ["project"]
            }),
        ),
    ]
}
