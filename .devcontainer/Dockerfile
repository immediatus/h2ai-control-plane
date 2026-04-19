FROM mcr.microsoft.com/devcontainers/rust:1-bookworm

# llama.cpp FFI build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    clang \
    libclang-dev \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cargo tools
RUN cargo install cargo-nextest --locked \
    && cargo install cargo-watch --locked \
    && cargo install cargo-expand --locked

# Python tooling for math validation and simulation scripts
RUN pip install --no-cache-dir numpy matplotlib

USER vscode
