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

# Bun (required for Claude plugins e.g. telegram MCP)
RUN curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash

# Python tooling for math validation and simulation scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages numpy matplotlib openai requests datasets

# Setup developer user by renaming default vscode user
RUN groupmod -n developer vscode \
    && usermod -l developer -d /home/developer -m vscode \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/developer \
    && chmod 0440 /etc/sudoers.d/developer \
    && chown -R developer /usr/local/cargo

USER developer
ENV USER=developer
ENV HOME=/home/developer
ENV PATH="/home/developer/.local/bin:${PATH}"
RUN mkdir -p /home/developer/.local/bin
