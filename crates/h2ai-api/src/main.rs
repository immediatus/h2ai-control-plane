mod error;
mod routes;
mod state;

use axum::Router;
use h2ai_config::H2AIConfig;
use h2ai_state::nats::NatsClient;
use state::AppState;
use std::env;

#[tokio::main]
async fn main() {
    let nats_url = env::var("H2AI_NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let listen_addr = env::var("H2AI_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());

    let nats = NatsClient::connect(&nats_url).await.expect("NATS connect");
    nats.ensure_infrastructure()
        .await
        .expect("NATS infrastructure setup");

    let cfg = H2AIConfig::default();
    let app_state = AppState::new(nats, cfg);

    let app = Router::new()
        .merge(routes::task_router())
        .merge(routes::calibrate_router())
        .merge(routes::health_router())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    eprintln!("listening on {listen_addr}");
    axum::serve(listener, app).await.unwrap();
}
