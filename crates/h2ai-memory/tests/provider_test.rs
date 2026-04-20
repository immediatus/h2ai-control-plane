use async_trait::async_trait;
use h2ai_memory::error::MemoryError;
use h2ai_memory::provider::MemoryProvider;
use serde_json::json;

struct MockMemory {
    entries: std::sync::Mutex<Vec<serde_json::Value>>,
}

impl MockMemory {
    fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(vec![]),
        }
    }
}

#[async_trait]
impl MemoryProvider for MockMemory {
    async fn get_recent_history(
        &self,
        _session_id: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, MemoryError> {
        let entries = self.entries.lock().unwrap();
        Ok(entries.iter().rev().take(limit).cloned().collect())
    }

    async fn commit_new_memories(
        &self,
        _session_id: &str,
        memories: Vec<serde_json::Value>,
    ) -> Result<(), MemoryError> {
        let mut entries = self.entries.lock().unwrap();
        entries.extend(memories);
        Ok(())
    }

    async fn retrieve_relevant_context(
        &self,
        _session_id: &str,
        _query: &str,
    ) -> Result<Vec<String>, MemoryError> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn memory_provider_commit_and_retrieve_history() {
    let mem = MockMemory::new();
    mem.commit_new_memories("s1", vec![json!({"role": "user", "content": "hello"})])
        .await
        .unwrap();
    let history = mem.get_recent_history("s1", 10).await.unwrap();
    assert_eq!(history.len(), 1);
}

#[tokio::test]
async fn memory_provider_limit_is_respected() {
    let mem = MockMemory::new();
    for i in 0..5 {
        mem.commit_new_memories("s1", vec![json!({"i": i})])
            .await
            .unwrap();
    }
    let history = mem.get_recent_history("s1", 3).await.unwrap();
    assert_eq!(history.len(), 3);
}

#[tokio::test]
async fn memory_provider_relevant_context_returns_vec() {
    let mem = MockMemory::new();
    let ctx = mem.retrieve_relevant_context("s1", "query").await.unwrap();
    assert!(ctx.is_empty());
}
