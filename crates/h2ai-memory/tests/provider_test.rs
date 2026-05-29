use h2ai_memory::error::MemoryError;
use h2ai_memory::provider::MemoryProvider;
use serde_json::json;
use std::sync::{Arc, Mutex};

mockall::mock! {
    pub MockMemory {}
    #[async_trait::async_trait]
    impl MemoryProvider for MockMemory {
        async fn get_recent_history(&self, session_id: &str, limit: usize) -> Result<Vec<serde_json::Value>, MemoryError>;
        async fn commit_new_memories(&self, session_id: &str, memories: Vec<serde_json::Value>) -> Result<(), MemoryError>;
        async fn retrieve_relevant_context(&self, session_id: &str, query: &str) -> Result<Vec<String>, MemoryError>;
    }
}

/// Build a stateful in-memory MockMemory backed by a shared vec.
fn make_mock_memory() -> (MockMockMemory, Arc<Mutex<Vec<serde_json::Value>>>) {
    let store: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let store_commit = store.clone();
    let store_history = store.clone();

    let mut m = MockMockMemory::new();
    m.expect_commit_new_memories()
        .returning(move |_, memories| {
            store_commit.lock().unwrap().extend(memories);
            Ok(())
        });
    m.expect_get_recent_history().returning(move |_, limit| {
        let entries = store_history.lock().unwrap();
        Ok(entries.iter().rev().take(limit).cloned().collect())
    });
    m.expect_retrieve_relevant_context()
        .returning(|_, _| Ok(vec![]));
    (m, store)
}

#[tokio::test]
async fn memory_provider_commit_and_retrieve_history() {
    let (mem, _) = make_mock_memory();
    mem.commit_new_memories("s1", vec![json!({"role": "user", "content": "hello"})])
        .await
        .unwrap();
    let history = mem.get_recent_history("s1", 10).await.unwrap();
    assert_eq!(history.len(), 1);
}

#[tokio::test]
async fn memory_provider_limit_is_respected() {
    let (mem, _) = make_mock_memory();
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
    let (mem, _) = make_mock_memory();
    let ctx = mem.retrieve_relevant_context("s1", "query").await.unwrap();
    assert!(ctx.is_empty());
}
