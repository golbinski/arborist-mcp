use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeLabel {
    Project,
    File,
    Namespace,
    Class,
    Struct,
    Function,
    Method,
    Template,
    Enum,
    Variable,
}

impl NodeLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Project => "Project",
            Self::File => "File",
            Self::Namespace => "Namespace",
            Self::Class => "Class",
            Self::Struct => "Struct",
            Self::Function => "Function",
            Self::Method => "Method",
            Self::Template => "Template",
            Self::Enum => "Enum",
            Self::Variable => "Variable",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Project" => Some(Self::Project),
            "File" => Some(Self::File),
            "Namespace" => Some(Self::Namespace),
            "Class" => Some(Self::Class),
            "Struct" => Some(Self::Struct),
            "Function" => Some(Self::Function),
            "Method" => Some(Self::Method),
            "Template" => Some(Self::Template),
            "Enum" => Some(Self::Enum),
            "Variable" => Some(Self::Variable),
            _ => None,
        }
    }

    pub fn one_hot_index(&self) -> usize {
        match self {
            Self::Project => 0,
            Self::File => 1,
            Self::Namespace => 2,
            Self::Class => 3,
            Self::Struct => 4,
            Self::Function => 5,
            Self::Method => 6,
            Self::Template => 7,
            Self::Enum => 8,
            Self::Variable => 9,
        }
    }

    pub const COUNT: usize = 10;
}

impl fmt::Display for NodeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    Contains,
    Defines,
    Calls,
    Includes,
    Inherits,
    Instantiates,
    Overrides,
    UsesType,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Contains => "CONTAINS",
            Self::Defines => "DEFINES",
            Self::Calls => "CALLS",
            Self::Includes => "INCLUDES",
            Self::Inherits => "INHERITS",
            Self::Instantiates => "INSTANTIATES",
            Self::Overrides => "OVERRIDES",
            Self::UsesType => "USES_TYPE",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "CONTAINS" => Some(Self::Contains),
            "DEFINES" => Some(Self::Defines),
            "CALLS" => Some(Self::Calls),
            "INCLUDES" => Some(Self::Includes),
            "INHERITS" => Some(Self::Inherits),
            "INSTANTIATES" => Some(Self::Instantiates),
            "OVERRIDES" => Some(Self::Overrides),
            "USES_TYPE" => Some(Self::UsesType),
            _ => None,
        }
    }

    pub fn weight(&self) -> f32 {
        match self {
            Self::Calls => 1.0,
            Self::Inherits => 1.2,
            Self::Overrides => 1.1,
            Self::Instantiates => 0.9,
            Self::UsesType => 0.8,
            Self::Defines => 0.7,
            Self::Contains => 0.5,
            Self::Includes => 0.6,
        }
    }

    pub const ALL: &'static [Self] = &[
        Self::Contains,
        Self::Defines,
        Self::Calls,
        Self::Includes,
        Self::Inherits,
        Self::Instantiates,
        Self::Overrides,
        Self::UsesType,
    ];
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: i64,
    pub project: String,
    pub label: NodeLabel,
    pub qualified_name: String,
    pub file_path: Option<String>,
    pub line_start: Option<i32>,
    pub line_end: Option<i32>,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source_id: i64,
    pub target_id: i64,
    pub edge_type: EdgeType,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInput {
    pub project: String,
    pub label: NodeLabel,
    pub qualified_name: String,
    pub file_path: Option<String>,
    pub line_start: Option<i32>,
    pub line_end: Option<i32>,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInput {
    pub source_id: i64,
    pub target_id: i64,
    pub edge_type: EdgeType,
    pub properties: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_label_roundtrip() {
        for label in [
            NodeLabel::Project,
            NodeLabel::File,
            NodeLabel::Namespace,
            NodeLabel::Class,
            NodeLabel::Struct,
            NodeLabel::Function,
            NodeLabel::Method,
            NodeLabel::Template,
            NodeLabel::Enum,
            NodeLabel::Variable,
        ] {
            assert_eq!(NodeLabel::from_str(label.as_str()), Some(label));
        }
        assert_eq!(NodeLabel::from_str("Unknown"), None);
    }

    #[test]
    fn node_label_one_hot_indices_unique() {
        let mut seen = std::collections::HashSet::new();
        for label in [
            NodeLabel::Project,
            NodeLabel::File,
            NodeLabel::Namespace,
            NodeLabel::Class,
            NodeLabel::Struct,
            NodeLabel::Function,
            NodeLabel::Method,
            NodeLabel::Template,
            NodeLabel::Enum,
            NodeLabel::Variable,
        ] {
            let idx = label.one_hot_index();
            assert!(idx < NodeLabel::COUNT, "index {} out of range", idx);
            assert!(seen.insert(idx), "duplicate one_hot_index {}", idx);
        }
    }

    #[test]
    fn edge_type_roundtrip() {
        for et in EdgeType::ALL {
            assert_eq!(EdgeType::from_str(et.as_str()), Some(*et));
        }
        assert_eq!(EdgeType::from_str("UNKNOWN"), None);
    }

    #[test]
    fn edge_type_weights_positive() {
        for et in EdgeType::ALL {
            assert!(et.weight() > 0.0, "{} has non-positive weight", et.as_str());
        }
    }
}
