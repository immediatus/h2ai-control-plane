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
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages numpy matplotlib

# Install Node and Claude Code natively
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Dynamically handle the host user
ARG USERNAME=vscode
RUN if [ "$USERNAME" != "vscode" ]; then \
    useradd -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME; \
fi
USER $USERNAME
ENV USER=$USERNAME
ENV HOME=/home/$USERNAME
