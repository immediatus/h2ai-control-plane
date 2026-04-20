use chrono::Utc;
use h2ai_state::journal::{EventJournal, InMemoryBackend};
use h2ai_types::events::{H2AIEvent, MergeResolvedEvent, ZeroSurvivalEvent};
use h2ai_types::identity::TaskId;

#[tokio::test]
async fn append_and_replay_preserves_order() {
    let journal = EventJournal::new(InMemoryBackend::new());
    let tid = TaskId::new();

    journal
        .append(H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
            task_id: tid.clone(),
            retry_count: 0,
            timestamp: Utc::now(),
        }))
        .await
        .unwrap();

    journal
        .append(H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "done".into(),
            timestamp: Utc::now(),
        }))
        .await
        .unwrap();

    let events = journal.replay(0).await.unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], H2AIEvent::ZeroSurvival(_)));
    assert!(matches!(events[1], H2AIEvent::MergeResolved(_)));
}

#[tokio::test]
async fn replay_from_offset_returns_tail() {
    let journal = EventJournal::new(InMemoryBackend::new());
    let tid = TaskId::new();

    for i in 0..5u32 {
        journal
            .append(H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
                task_id: tid.clone(),
                retry_count: i,
                timestamp: Utc::now(),
            }))
            .await
            .unwrap();
    }

    let tail = journal.replay(3).await.unwrap();
    assert_eq!(tail.len(), 2);
}

#[tokio::test]
async fn replay_empty_journal_returns_empty_vec() {
    let journal = EventJournal::new(InMemoryBackend::new());
    let events = journal.replay(0).await.unwrap();
    assert!(events.is_empty());
}
