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
    && pip3 install --no-cache-dir --break-system-packages numpy matplotlib

# Rename vscode user/group to match host identity; adjust UID/GID if needed for keep-id
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN if [ "$USERNAME" != "vscode" ]; then \
        groupmod -n "$USERNAME" vscode \
        && usermod -l "$USERNAME" -d "/home/$USERNAME" -m vscode \
        && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"$USERNAME" \
        && chmod 0440 /etc/sudoers.d/"$USERNAME"; \
    fi \
    && if [ "$USER_UID" != "$(id -u $USERNAME)" ]; then usermod -u "$USER_UID" "$USERNAME"; fi \
    && if [ "$USER_GID" != "$(id -g $USERNAME)" ]; then groupmod -g "$USER_GID" "$USERNAME"; fi \
    && chown -R "$USERNAME" /usr/local/cargo
USER $USERNAME
ENV USER=$USERNAME
ENV HOME=/home/$USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
