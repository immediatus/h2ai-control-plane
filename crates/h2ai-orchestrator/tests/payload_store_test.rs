use h2ai_orchestrator::payload_store::{
    offload_if_large, resolve_context, MemoryPayloadStore, PayloadStore, StoreError,
};
use h2ai_types::agent::ContextPayload;

#[tokio::test]
async fn memory_store_roundtrip() {
    let store = MemoryPayloadStore::new();
    let hash = store.put("hello world").await.unwrap();
    let retrieved = store.get(&hash).await.unwrap();
    assert_eq!(retrieved, "hello world");
}

#[tokio::test]
async fn memory_store_deduplication() {
    let store = MemoryPayloadStore::new();
    let h1 = store.put("same content").await.unwrap();
    let h2 = store.put("same content").await.unwrap();
    assert_eq!(h1, h2, "identical content must produce identical hash");
}

#[tokio::test]
async fn memory_store_not_found_returns_error() {
    let store = MemoryPayloadStore::new();
    let unknown = [0u8; 32];
    let result = store.get(&unknown).await;
    assert!(matches!(result, Err(StoreError::NotFound)));
}

#[tokio::test]
async fn inline_payload_resolves_without_store_call() {
    let store = MemoryPayloadStore::new();
    let payload = ContextPayload::Inline("tiny".into());
    let resolved = resolve_context(&payload, &store).await.unwrap();
    assert_eq!(resolved, "tiny");
    // Store must be empty — no put was called.
    let any_hash = [0u8; 32];
    assert!(matches!(
        store.get(&any_hash).await,
        Err(StoreError::NotFound)
    ));
}

#[tokio::test]
async fn offload_below_threshold_stays_inline() {
    let store = MemoryPayloadStore::new();
    let result = offload_if_large("small".into(), 100, &store).await.unwrap();
    assert!(matches!(result, ContextPayload::Inline(s) if s == "small"));
}

#[tokio::test]
async fn offload_above_threshold_produces_ref() {
    let store = MemoryPayloadStore::new();
    let large = "x".repeat(200);
    let result = offload_if_large(large.clone(), 100, &store).await.unwrap();
    match result {
        ContextPayload::Ref { hash, byte_len } => {
            assert_eq!(byte_len, 200);
            assert_eq!(hash.len(), 64, "hex-encoded SHA-256 must be 64 chars");
            // Resolve back to original content.
            let resolved = resolve_context(&ContextPayload::Ref { hash, byte_len }, &store)
                .await
                .unwrap();
            assert_eq!(resolved, large);
        }
        ContextPayload::Inline(_) => panic!("expected Ref, got Inline"),
    }
}

#[tokio::test]
async fn two_large_offloads_of_same_content_share_one_store_entry() {
    let store = std::sync::Arc::new(MemoryPayloadStore::new());
    let content = "y".repeat(300);
    let r1 = offload_if_large(content.clone(), 100, store.as_ref())
        .await
        .unwrap();
    let r2 = offload_if_large(content.clone(), 100, store.as_ref())
        .await
        .unwrap();
    match (&r1, &r2) {
        (ContextPayload::Ref { hash: h1, .. }, ContextPayload::Ref { hash: h2, .. }) => {
            assert_eq!(
                h1, h2,
                "identical large content must produce identical hash"
            );
        }
        _ => panic!("both should be Ref"),
    }
}

#[tokio::test]
async fn resolve_ref_with_unknown_hash_returns_not_found() {
    let store = MemoryPayloadStore::new();
    let bad_ref = ContextPayload::Ref {
        hash: "0".repeat(64),
        byte_len: 0,
    };
    let result = resolve_context(&bad_ref, &store).await;
    assert!(matches!(result, Err(StoreError::NotFound)));
}
