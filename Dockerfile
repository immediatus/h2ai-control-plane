# ── builder ───────────────────────────────────────────────────────────────────
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    clang \
    libclang-dev \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependencies before copying source
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/
RUN cargo build --release --locked --bin h2ai-control-plane

# ── runtime ───────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/h2ai-control-plane /usr/local/bin/h2ai-control-plane

RUN useradd --system --no-create-home h2ai
USER h2ai

EXPOSE 8080 9090

ENTRYPOINT ["h2ai-control-plane"]
