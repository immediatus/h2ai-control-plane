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
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    #[must_use]
    pub const fn from_uuid(u: Uuid) -> Self {
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
    #[must_use]
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
    #[must_use]
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

/// Tenant identifier — scope boundary for per-tenant KV buckets and task meta-state.
///
/// Defaults to `"default"` for single-tenant deployments (backward compatible).
/// Bucket names are derived as `{prefix}_{sanitized_tenant_id}` where
/// non-alphanumeric characters are replaced with `_`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl TenantId {
    #[must_use]
    pub fn default_tenant() -> Self {
        Self("default".into())
    }

    /// Returns the tenant id sanitized for use in NATS KV bucket names.
    /// Replaces hyphens, dots, and spaces with underscores.
    #[must_use]
    pub fn bucket_safe(&self) -> String {
        self.0
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TenantId {
    fn default() -> Self {
        Self::default_tenant()
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl AsRef<str> for TenantId {
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}
