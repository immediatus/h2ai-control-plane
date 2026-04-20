use h2ai_memory::in_memory::InMemoryCache;
use h2ai_memory::provider::MemoryProvider;
use serde_json::json;

#[tokio::test]
async fn in_memory_cache_stores_and_retrieves() {
    let cache = InMemoryCache::new();
    cache
        .commit_new_memories("s1", vec![json!({"msg": "hello"})])
        .await
        .unwrap();
    let history = cache.get_recent_history("s1", 10).await.unwrap();
    assert_eq!(history.len(), 1);
}

#[tokio::test]
async fn in_memory_cache_limit_is_respected() {
    let cache = InMemoryCache::new();
    for i in 0..10 {
        cache
            .commit_new_memories("s1", vec![json!({"i": i})])
            .await
            .unwrap();
    }
    let history = cache.get_recent_history("s1", 3).await.unwrap();
    assert_eq!(history.len(), 3);
}

#[tokio::test]
async fn in_memory_cache_empty_session_returns_empty() {
    let cache = InMemoryCache::new();
    let history = cache.get_recent_history("nonexistent", 10).await.unwrap();
    assert!(history.is_empty());
}

#[tokio::test]
async fn in_memory_cache_multiple_sessions_isolated() {
    let cache = InMemoryCache::new();
    cache
        .commit_new_memories("s1", vec![json!({"a": 1})])
        .await
        .unwrap();
    cache
        .commit_new_memories("s2", vec![json!({"b": 2}), json!({"c": 3})])
        .await
        .unwrap();
    let h1 = cache.get_recent_history("s1", 10).await.unwrap();
    let h2 = cache.get_recent_history("s2", 10).await.unwrap();
    assert_eq!(h1.len(), 1);
    assert_eq!(h2.len(), 2);
}
