use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::IntoResponse,
};

/// Legacy endpoint — redirects to `POST /signal`.
pub async fn approve_task(Path((tenant_id, task_id)): Path<(String, String)>) -> impl IntoResponse {
    let location = format!("/v1/{tenant_id}/tasks/{task_id}/signal");
    (
        StatusCode::PERMANENT_REDIRECT,
        [(header::LOCATION, location)],
    )
}

/// Legacy GET endpoint — return 410 Gone.
pub async fn get_approval(
    Path((_tenant_id, task_id)): Path<(String, String)>,
) -> impl IntoResponse {
    (
        StatusCode::GONE,
        format!("approval records removed; task {task_id} uses /signal endpoint"),
    )
}
