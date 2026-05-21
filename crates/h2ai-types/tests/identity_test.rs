use h2ai_types::identity::{AgentId, ExplorerId, SubtaskId, TaskId, TenantId};

#[test]
fn task_id_is_unique_each_time() {
    let a = TaskId::new();
    let b = TaskId::new();
    assert_ne!(a, b);
}

#[test]
fn task_id_display_is_hyphenated_uuid() {
    let id = TaskId::new();
    let s = id.to_string();
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
}

#[test]
fn explorer_id_is_unique_each_time() {
    let a = ExplorerId::new();
    let b = ExplorerId::new();
    assert_ne!(a, b);
}

#[test]
fn task_id_and_explorer_id_are_distinct_types() {
    let _t = TaskId::new();
    let _e = ExplorerId::new();
}

#[test]
fn task_id_serde_round_trip() {
    let id = TaskId::new();
    let json = serde_json::to_string(&id).unwrap();
    let back: TaskId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}

#[test]
fn explorer_id_serde_round_trip() {
    let id = ExplorerId::new();
    let json = serde_json::to_string(&id).unwrap();
    let back: ExplorerId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}

#[test]
fn agent_id_from_str_and_display() {
    let id: AgentId = "agent-42".into();
    assert_eq!(id.to_string(), "agent-42");
}

#[test]
fn agent_id_from_string() {
    let id: AgentId = String::from("agent-7").into();
    assert_eq!(id.to_string(), "agent-7");
}

#[test]
fn agent_id_serde_round_trip() {
    let id: AgentId = "agent-99".into();
    let json = serde_json::to_string(&id).unwrap();
    let back: AgentId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}

// ── AgentId::as_ref ───────────────────────────────────────────────────────────

#[test]
fn agent_id_as_ref_returns_inner_str() {
    let id: AgentId = "my-agent".into();
    let s: &str = id.as_ref();
    assert_eq!(s, "my-agent");
}

// ── TaskId::default and from_uuid ────────────────────────────────────────────

#[test]
fn task_id_default_is_new() {
    let a = TaskId::default();
    let b = TaskId::default();
    assert_ne!(a, b, "default() should produce unique UUIDs");
}

#[test]
fn task_id_from_uuid_round_trips_display() {
    let u = uuid::Uuid::new_v4();
    let id = TaskId::from_uuid(u);
    assert_eq!(id.to_string(), u.to_string());
}

// ── ExplorerId default and display ───────────────────────────────────────────

#[test]
fn explorer_id_default_is_new() {
    let a = ExplorerId::default();
    let b = ExplorerId::default();
    assert_ne!(a, b);
}

#[test]
fn explorer_id_display_is_hyphenated_uuid() {
    let id = ExplorerId::new();
    let s = id.to_string();
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
}

// ── SubtaskId ─────────────────────────────────────────────────────────────────

#[test]
fn subtask_id_new_is_unique() {
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    assert_ne!(a, b);
}

#[test]
fn subtask_id_default_is_new() {
    let a = SubtaskId::default();
    let b = SubtaskId::default();
    assert_ne!(a, b);
}

#[test]
fn subtask_id_display_is_hyphenated_uuid() {
    let id = SubtaskId::new();
    let s = id.to_string();
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
}

#[test]
fn subtask_id_serde_round_trip() {
    let id = SubtaskId::new();
    let json = serde_json::to_string(&id).unwrap();
    let back: SubtaskId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}

// ── TenantId ──────────────────────────────────────────────────────────────────

#[test]
fn tenant_id_default_is_default_tenant() {
    let id = TenantId::default();
    assert_eq!(id.to_string(), "default");
}

#[test]
fn tenant_id_default_tenant_constructor() {
    let id = TenantId::default_tenant();
    assert_eq!(id.to_string(), "default");
}

#[test]
fn tenant_id_from_str() {
    let id: TenantId = "payments-team".into();
    assert_eq!(id.to_string(), "payments-team");
}

#[test]
fn tenant_id_from_string() {
    let id: TenantId = String::from("acme-corp").into();
    assert_eq!(id.to_string(), "acme-corp");
}

#[test]
fn tenant_id_as_ref() {
    let id: TenantId = "my-tenant".into();
    let s: &str = id.as_ref();
    assert_eq!(s, "my-tenant");
}

#[test]
fn tenant_id_display() {
    let id: TenantId = "team-alpha".into();
    assert_eq!(format!("{id}"), "team-alpha");
}

#[test]
fn tenant_id_bucket_safe_alphanumeric_unchanged() {
    let id: TenantId = "team_alpha".into();
    assert_eq!(id.bucket_safe(), "team_alpha");
}

#[test]
fn tenant_id_bucket_safe_replaces_hyphens_dots_spaces() {
    let id: TenantId = "my-team.name here".into();
    assert_eq!(id.bucket_safe(), "my_team_name_here");
}

#[test]
fn tenant_id_serde_round_trip() {
    let id: TenantId = "eu-west".into();
    let json = serde_json::to_string(&id).unwrap();
    let back: TenantId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}
