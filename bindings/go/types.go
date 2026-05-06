// Hand-authored Go bindings for h2ai-types.
// typeshare-cli dropped Go language support in v1.13+; these must be maintained manually.
// Last updated: 2026-05-05

package h2aitypes

import "time"

type AgentTool string

const (
	AgentToolShell         AgentTool = "Shell"
	AgentToolWebSearch     AgentTool = "WebSearch"
	AgentToolCodeExecution AgentTool = "CodeExecution"
	AgentToolFileSystem    AgentTool = "FileSystem"
)

// Ordered Low < Mid < High. Agents declare their tier; tasks declare a maximum.
type CostTier string

const (
	CostTierLow  CostTier = "Low"
	CostTierMid  CostTier = "Mid"
	CostTierHigh CostTier = "High"
)

// Scheduling requirements a task passes to AgentProvider::select_agent.
type TaskRequirements struct {
	MaxCostTier   CostTier    `json:"max_cost_tier"`
	RequiredTools []AgentTool `json:"required_tools"`
}

type AgentDescriptor struct {
	Model    string      `json:"model"`
	Tools    []AgentTool `json:"tools"`
	CostTier CostTier    `json:"cost_tier"`
}

type AgentState struct {
	State   string  `json:"state"`
	Message *string `json:"message,omitempty"`
}

// System context carried in a NATS task message — either inlined (small payloads) or
// referenced by SHA-256 hex hash in a content-addressed object store (large payloads).
type ContextPayload struct {
	Kind  string      `json:"kind"`
	Value interface{} `json:"value"`
}

// ContextPayloadInline is the inline variant of ContextPayload.
type ContextPayloadInline struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

// ContextPayloadRef is the content-addressed reference variant of ContextPayload.
type ContextPayloadRef struct {
	Kind  string `json:"kind"`
	Value struct {
		Hash    string `json:"hash"`
		ByteLen int    `json:"byte_len"`
	} `json:"value"`
}

type WaveMode string

const (
	WaveModeNormal   WaveMode = "Normal"
	WaveModeHardened WaveMode = "Hardened"
)

type TaskPayload struct {
	TaskID       string          `json:"task_id"`
	AgentID      string          `json:"agent_id"`
	Agent        AgentDescriptor `json:"agent"`
	Instructions string          `json:"instructions"`
	Context      ContextPayload  `json:"context"`
	Tau          float64         `json:"tau"`
	MaxTokens    uint64          `json:"max_tokens"`
	WaveMode     WaveMode        `json:"wave_mode"`
}

type TaskResult struct {
	TaskID     string  `json:"task_id"`
	AgentID    string  `json:"agent_id"`
	Output     string  `json:"output"`
	TokenCost  uint64  `json:"token_cost"`
	Error      *string `json:"error,omitempty"`
}

type AgentHeartbeat struct {
	AgentID     string          `json:"agent_id"`
	Descriptor  AgentDescriptor `json:"descriptor"`
	Timestamp   time.Time       `json:"timestamp"`
	ActiveTasks uint32          `json:"active_tasks"`
}

type AgentTelemetryEventLlmPromptSent struct {
	EventType string    `json:"event_type"`
	TaskID    string    `json:"task_id"`
	AgentID   string    `json:"agent_id"`
	Prompt    string    `json:"prompt"`
	Timestamp time.Time `json:"timestamp"`
}

type AgentTelemetryEventLlmResponseReceived struct {
	EventType string    `json:"event_type"`
	TaskID    string    `json:"task_id"`
	AgentID   string    `json:"agent_id"`
	Response  string    `json:"response"`
	TokenCost uint64    `json:"token_cost"`
	Timestamp time.Time `json:"timestamp"`
}

type AgentTelemetryEventShellCommandExecuted struct {
	EventType string    `json:"event_type"`
	TaskID    string    `json:"task_id"`
	AgentID   string    `json:"agent_id"`
	Command   string    `json:"command"`
	Args      []string  `json:"args"`
	Stdout    string    `json:"stdout"`
	Stderr    string    `json:"stderr"`
	ExitCode  int32     `json:"exit_code"`
	Timestamp time.Time `json:"timestamp"`
}

type AgentTelemetryEventSystemError struct {
	EventType string    `json:"event_type"`
	TaskID    string    `json:"task_id"`
	AgentID   string    `json:"agent_id"`
	Error     string    `json:"error"`
	Timestamp time.Time `json:"timestamp"`
}
