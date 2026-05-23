use crate::{
    source::{ConstraintError, ConstraintSource},
    spec::SemanticSpec,
    versioned::{RepairProvenance, VersionConflictError, VersionedConstraintSource, VersionedSpec},
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

type OverrideEntry = (SemanticSpec, Option<RepairProvenance>);

/// Versioned constraint source that wraps any `ConstraintSource` with an in-memory
/// override map and optional NATS JetStream KV persistence.
///
/// In-memory mode (`new_in_memory`) is suitable for tests and single-node deployments.
/// NATS mode (`new_nats`) persists versions to a KV bucket for multi-node coordination.
pub struct NatsVersionedSource<S: ConstraintSource> {
    inner: S,
    overrides: Arc<RwLock<HashMap<String, OverrideEntry>>>,
    kv: Option<async_nats::jetstream::kv::Store>,
}

impl<S: ConstraintSource> NatsVersionedSource<S> {
    /// Create a source backed only by an in-memory override map (no NATS connection required).
    pub fn new_in_memory(inner: S) -> Self {
        Self {
            inner,
            overrides: Arc::new(RwLock::new(HashMap::new())),
            kv: None,
        }
    }

    /// Create a source backed by both in-memory cache and a NATS JetStream KV store.
    pub fn new_nats(inner: S, kv: async_nats::jetstream::kv::Store) -> Self {
        Self {
            inner,
            overrides: Arc::new(RwLock::new(HashMap::new())),
            kv: Some(kv),
        }
    }

    fn version_key(id: &str, version: u64) -> String {
        format!("h2ai.constraints.{id}.v{version}")
    }

    fn latest_key(id: &str) -> String {
        format!("h2ai.constraints.{id}.latest")
    }
}

impl<S: ConstraintSource> ConstraintSource for NatsVersionedSource<S> {
    fn load_all(&self) -> Result<Vec<SemanticSpec>, ConstraintError> {
        let mut specs = self.inner.load_all()?;
        let lock = self.overrides.read().unwrap();
        for spec in &mut specs {
            if let Some((override_spec, _)) = lock.get(&spec.id) {
                *spec = override_spec.clone();
            }
        }
        Ok(specs)
    }
}

#[async_trait]
impl<S: ConstraintSource> VersionedConstraintSource for NatsVersionedSource<S> {
    async fn load_latest_versioned(&self, id: &str) -> Result<VersionedSpec, ConstraintError> {
        // Check in-memory overrides first
        {
            let lock = self.overrides.read().unwrap();
            if let Some((spec, provenance)) = lock.get(id) {
                return Ok(VersionedSpec {
                    spec: spec.clone(),
                    provenance: provenance.clone(),
                });
            }
        }

        // Try NATS KV if available
        if let Some(kv) = &self.kv {
            let latest_key = Self::latest_key(id);
            if let Ok(Some(entry)) = kv.entry(&latest_key).await {
                if let Ok(version_str) = std::str::from_utf8(&entry.value) {
                    if let Ok(version) = version_str.trim().parse::<u64>() {
                        let version_key = Self::version_key(id, version);
                        if let Ok(Some(vs_entry)) = kv.entry(&version_key).await {
                            if let Ok(vs) = serde_json::from_slice::<VersionedSpec>(&vs_entry.value)
                            {
                                // Warm the in-memory cache
                                self.overrides.write().unwrap().insert(
                                    id.to_owned(),
                                    (vs.spec.clone(), vs.provenance.clone()),
                                );
                                return Ok(vs);
                            }
                        }
                    }
                }
            }
        }

        // Fall through to inner
        let specs = self.inner.load_all()?;
        let spec = specs
            .into_iter()
            .find(|s| s.id == id)
            .ok_or_else(|| ConstraintError::NotFound(id.to_owned()))?;
        Ok(VersionedSpec {
            spec,
            provenance: None,
        })
    }

    /// Note: `spec.version` is overwritten with `expected_version + 1` regardless
    /// of the value passed in.
    async fn create_next_version(
        &self,
        id: &str,
        expected_version: u64,
        mut spec: SemanticSpec,
        provenance: RepairProvenance,
    ) -> Result<u64, VersionConflictError> {
        let new_version = expected_version + 1;
        spec.version = new_version;
        spec.repair_provenance = Some(provenance.clone());

        // Acquire write lock once — do CAS check and write atomically
        {
            let mut lock = self.overrides.write().unwrap();
            let current_version = lock.get(id).map(|(s, _)| s.version).unwrap_or_else(|| {
                self.inner
                    .load_all()
                    .ok()
                    .and_then(|specs| specs.into_iter().find(|s| s.id == id))
                    .map(|s| s.version)
                    .unwrap_or(1)
            });

            if current_version != expected_version {
                return Err(VersionConflictError {
                    constraint_id: id.to_owned(),
                    expected: expected_version,
                    actual: current_version,
                });
            }

            lock.insert(id.to_owned(), (spec.clone(), Some(provenance.clone())));
        } // write lock dropped before any async work

        // Persist to NATS KV if available (best-effort, lock already released)
        if let Some(kv) = &self.kv {
            let vs = VersionedSpec {
                spec: spec.clone(),
                provenance: Some(provenance),
            };
            if let Ok(version_bytes) = serde_json::to_vec(&vs) {
                let version_key = Self::version_key(id, new_version);
                let latest_key = Self::latest_key(id);
                let _ = kv.put(&version_key, version_bytes.into()).await;
                let _ = kv
                    .put(&latest_key, new_version.to_string().into_bytes().into())
                    .await;
            }
        }

        Ok(new_version)
    }
}
