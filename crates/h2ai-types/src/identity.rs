use serde::{Deserialize, Serialize};
use std::fmt;
use typeshare::typeshare;
use uuid::Uuid;

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for AgentId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for AgentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl AsRef<str> for AgentId {
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(Uuid);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExplorerId(Uuid);

impl ExplorerId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ExplorerId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExplorerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubtaskId(Uuid);

impl SubtaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SubtaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SubtaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
