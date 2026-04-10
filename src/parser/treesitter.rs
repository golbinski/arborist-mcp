use anyhow::Result;
use std::path::Path;
use tree_sitter::{Language, Node, Parser};

use crate::graph::schema::NodeLabel;

/// Extracted symbol from a single C++ file.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub label: NodeLabel,
    pub qualified_name: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub parent_scope: Option<String>,
}

/// Call-site extracted during tree-sitter pass.
#[derive(Debug, Clone)]
pub struct CallSite {
    pub caller_qname: String,
    pub callee_name: String,
    pub file_path: String,
    pub line: u32,
}

/// Include directive.
#[derive(Debug, Clone)]
pub struct IncludeDirective {
    pub from_file: String,
    pub included_path: String,
}

pub struct TreeSitterExtractor {
    parser: Parser,
    lang: Language,
}

impl TreeSitterExtractor {
    pub fn new() -> Result<Self> {
        let lang: Language = tree_sitter_cpp::LANGUAGE.into();
        let mut parser = Parser::new();
        parser.set_language(&lang)?;
        Ok(Self { parser, lang })
    }

    pub fn extract_file(
        &mut self,
        file_path: &Path,
        source: &[u8],
    ) -> Result<(Vec<Symbol>, Vec<CallSite>, Vec<IncludeDirective>)> {
        let tree = match self.parser.parse(source, None) {
            Some(t) => t,
            None => return Ok((vec![], vec![], vec![])),
        };

        let root = tree.root_node();
        let file_str = file_path.to_string_lossy().into_owned();

        let mut symbols = Vec::new();
        let mut calls = Vec::new();
        let mut includes = Vec::new();

        self.walk_node(
            root,
            source,
            &file_str,
            &mut Vec::new(),
            &mut symbols,
            &mut calls,
            &mut includes,
        );

        Ok((symbols, calls, includes))
    }

    fn walk_node(
        &self,
        node: Node,
        source: &[u8],
        file_path: &str,
        scope_stack: &mut Vec<String>,
        symbols: &mut Vec<Symbol>,
        calls: &mut Vec<CallSite>,
        includes: &mut Vec<IncludeDirective>,
    ) {
        match node.kind() {
            "preproc_include" => {
                if let Some(path_node) = node
                    .child_by_field_name("path")
                    .or_else(|| node.named_child(0))
                {
                    let raw = node_text(path_node, source);
                    let included = raw.trim_matches(|c| c == '"' || c == '<' || c == '>').to_owned();
                    includes.push(IncludeDirective {
                        from_file: file_path.to_owned(),
                        included_path: included,
                    });
                }
            }

            "namespace_definition" => {
                let name = child_text_by_field(node, "name", source)
                    .unwrap_or_else(|| "(anonymous)".to_owned());
                let qname = qualify(&scope_stack, &name);
                let (ls, le) = node_lines(node);
                symbols.push(Symbol {
                    label: NodeLabel::Namespace,
                    qualified_name: qname.clone(),
                    file_path: file_path.to_owned(),
                    line_start: ls,
                    line_end: le,
                    parent_scope: scope_stack.last().cloned(),
                });
                scope_stack.push(qname);
                self.walk_children(node, source, file_path, scope_stack, symbols, calls, includes);
                scope_stack.pop();
                return;
            }

            "class_specifier" | "struct_specifier" => {
                let label = if node.kind() == "class_specifier" {
                    NodeLabel::Class
                } else {
                    NodeLabel::Struct
                };
                let name = child_text_by_field(node, "name", source)
                    .unwrap_or_else(|| "(anonymous)".to_owned());
                let qname = qualify(&scope_stack, &name);
                let (ls, le) = node_lines(node);
                symbols.push(Symbol {
                    label,
                    qualified_name: qname.clone(),
                    file_path: file_path.to_owned(),
                    line_start: ls,
                    line_end: le,
                    parent_scope: scope_stack.last().cloned(),
                });
                // Extract base classes
                if let Some(bases) = node.child_by_field_name("bases") {
                    for i in 0..bases.child_count() {
                        if let Some(base) = bases.child(i) {
                            if base.kind() == "base_class_clause" || base.kind() == "type_identifier" {
                                // Handled separately in the call resolution pass
                            }
                        }
                    }
                }
                scope_stack.push(qname);
                self.walk_children(node, source, file_path, scope_stack, symbols, calls, includes);
                scope_stack.pop();
                return;
            }

            "template_declaration" => {
                // Peek inside to find what's being templated
                let inner = node.child_by_field_name("declaration");
                if let Some(inner) = inner {
                    match inner.kind() {
                        "function_definition" | "declaration" => {
                            // Will be extracted as Template node
                            let name = extract_function_name(inner, source)
                                .unwrap_or_else(|| "(template)".to_owned());
                            let qname = qualify(&scope_stack, &name);
                            let (ls, le) = node_lines(node);
                            symbols.push(Symbol {
                                label: NodeLabel::Template,
                                qualified_name: qname.clone(),
                                file_path: file_path.to_owned(),
                                line_start: ls,
                                line_end: le,
                                parent_scope: scope_stack.last().cloned(),
                            });
                            // Don't recurse — avoid double-extracting the body
                            return;
                        }
                        "class_specifier" | "struct_specifier" => {
                            let name = child_text_by_field(inner, "name", source)
                                .unwrap_or_else(|| "(template)".to_owned());
                            let qname = qualify(&scope_stack, &name);
                            let (ls, le) = node_lines(node);
                            symbols.push(Symbol {
                                label: NodeLabel::Template,
                                qualified_name: qname.clone(),
                                file_path: file_path.to_owned(),
                                line_start: ls,
                                line_end: le,
                                parent_scope: scope_stack.last().cloned(),
                            });
                            return;
                        }
                        _ => {}
                    }
                }
            }

            "function_definition" => {
                let label = if scope_stack.last().map(|_| true).unwrap_or(false) {
                    // Heuristic: if current scope is a class/struct, it's a method
                    NodeLabel::Method
                } else {
                    NodeLabel::Function
                };
                if let Some(name) = extract_function_name(node, source) {
                    let qname = qualify(&scope_stack, &name);
                    let (ls, le) = node_lines(node);
                    symbols.push(Symbol {
                        label,
                        qualified_name: qname.clone(),
                        file_path: file_path.to_owned(),
                        line_start: ls,
                        line_end: le,
                        parent_scope: scope_stack.last().cloned(),
                    });
                    // Walk body for call sites
                    scope_stack.push(qname.clone());
                    if let Some(body) = node.child_by_field_name("body") {
                        self.collect_calls(body, source, file_path, &qname, calls);
                    }
                    scope_stack.pop();
                    return;
                }
            }

            "enum_specifier" => {
                let name = child_text_by_field(node, "name", source)
                    .unwrap_or_else(|| "(anonymous)".to_owned());
                let qname = qualify(&scope_stack, &name);
                let (ls, le) = node_lines(node);
                symbols.push(Symbol {
                    label: NodeLabel::Enum,
                    qualified_name: qname,
                    file_path: file_path.to_owned(),
                    line_start: ls,
                    line_end: le,
                    parent_scope: scope_stack.last().cloned(),
                });
            }

            "declaration" => {
                // Top-level variable declarations
                if scope_stack.is_empty() {
                    if let Some(decl) = node.child_by_field_name("declarator") {
                        let name = simple_declarator_name(decl, source);
                        if !name.is_empty() {
                            let qname = qualify(&scope_stack, &name);
                            let (ls, le) = node_lines(node);
                            symbols.push(Symbol {
                                label: NodeLabel::Variable,
                                qualified_name: qname,
                                file_path: file_path.to_owned(),
                                line_start: ls,
                                line_end: le,
                                parent_scope: None,
                            });
                        }
                    }
                }
            }

            _ => {}
        }

        self.walk_children(node, source, file_path, scope_stack, symbols, calls, includes);
    }

    fn walk_children(
        &self,
        node: Node,
        source: &[u8],
        file_path: &str,
        scope_stack: &mut Vec<String>,
        symbols: &mut Vec<Symbol>,
        calls: &mut Vec<CallSite>,
        includes: &mut Vec<IncludeDirective>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk_node(child, source, file_path, scope_stack, symbols, calls, includes);
        }
    }

    fn collect_calls(
        &self,
        node: Node,
        source: &[u8],
        file_path: &str,
        caller: &str,
        calls: &mut Vec<CallSite>,
    ) {
        if node.kind() == "call_expression" {
            if let Some(func) = node.child_by_field_name("function") {
                let name = call_target_name(func, source);
                if !name.is_empty() && name != caller {
                    calls.push(CallSite {
                        caller_qname: caller.to_owned(),
                        callee_name: name,
                        file_path: file_path.to_owned(),
                        line: node.start_position().row as u32 + 1,
                    });
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.collect_calls(child, source, file_path, caller, calls);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn node_text<'a>(node: Node, source: &'a [u8]) -> &'a str {
    node.utf8_text(source).unwrap_or("")
}

fn child_text_by_field(node: Node, field: &str, source: &[u8]) -> Option<String> {
    node.child_by_field_name(field)
        .map(|n| node_text(n, source).to_owned())
}

fn node_lines(node: Node) -> (u32, u32) {
    (
        node.start_position().row as u32 + 1,
        node.end_position().row as u32 + 1,
    )
}

fn qualify(stack: &[String], name: &str) -> String {
    if stack.is_empty() {
        name.to_owned()
    } else {
        format!("{}::{}", stack.last().unwrap(), name)
    }
}

fn extract_function_name(node: Node, source: &[u8]) -> Option<String> {
    // Try declarator → look for function_declarator → identifier / qualified_identifier
    let decl = node.child_by_field_name("declarator")?;
    Some(declarator_name(decl, source))
}

fn declarator_name(node: Node, source: &[u8]) -> String {
    match node.kind() {
        "identifier" | "type_identifier" => node_text(node, source).to_owned(),
        "qualified_identifier" => {
            // scope::name
            node_text(node, source).to_owned()
        }
        "destructor_name" => node_text(node, source).to_owned(),
        "operator_name" => node_text(node, source).to_owned(),
        "function_declarator" => {
            if let Some(decl) = node.child_by_field_name("declarator") {
                declarator_name(decl, source)
            } else {
                String::new()
            }
        }
        "pointer_declarator" | "reference_declarator" => {
            if let Some(inner) = node.named_child(0) {
                declarator_name(inner, source)
            } else {
                String::new()
            }
        }
        _ => {
            // Try first named child
            if let Some(child) = node.named_child(0) {
                declarator_name(child, source)
            } else {
                String::new()
            }
        }
    }
}

fn simple_declarator_name(node: Node, source: &[u8]) -> String {
    match node.kind() {
        "identifier" => node_text(node, source).to_owned(),
        "init_declarator" => {
            if let Some(d) = node.child_by_field_name("declarator") {
                simple_declarator_name(d, source)
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

fn call_target_name(node: Node, source: &[u8]) -> String {
    match node.kind() {
        "identifier" => node_text(node, source).to_owned(),
        "qualified_identifier" => {
            // Return just the final component
            if let Some(name) = node.child_by_field_name("name") {
                node_text(name, source).to_owned()
            } else {
                node_text(node, source).to_owned()
            }
        }
        "field_expression" => {
            // obj.method or obj->method
            if let Some(field) = node.child_by_field_name("field") {
                node_text(field, source).to_owned()
            } else {
                String::new()
            }
        }
        "template_function" => {
            if let Some(name) = node.child_by_field_name("name") {
                node_text(name, source).to_owned()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}
