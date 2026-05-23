use crate::error::ToolError;
use crate::{ToolExecutor, ToolSchema};
use async_trait::async_trait;
use serde_json::json;

#[async_trait]
pub trait WasmBackend: Send + Sync {
    async fn execute_script(&self, language: &str, script: &str) -> Result<String, ToolError>;
}

// ── Live: wasmtime interpreter sandbox ───────────────────────────────────────

#[cfg(feature = "wasm")]
pub struct RealWasmBackend {
    engine: wasmtime::Engine,
    module: wasmtime::Module,
    fuel_budget: u64,
}

#[cfg(feature = "wasm")]
impl RealWasmBackend {
    /// Load a WASM module from a file.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::InitializationFailed` if the file cannot be read, the
    /// Wasmtime engine cannot be configured, or the module cannot be compiled.
    pub fn from_file(path: &str, fuel_budget: u64) -> Result<Self, ToolError> {
        let wasm_bytes = std::fs::read(path)
            .map_err(|e| ToolError::InitializationFailed(format!("cannot read {path}: {e}")))?;

        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        config.wasm_multi_value(true);

        let engine = wasmtime::Engine::new(&config)
            .map_err(|e| ToolError::InitializationFailed(e.to_string()))?;
        let module = wasmtime::Module::new(&engine, &wasm_bytes)
            .map_err(|e| ToolError::InitializationFailed(e.to_string()))?;

        Ok(Self {
            engine,
            module,
            fuel_budget,
        })
    }
}

#[cfg(feature = "wasm")]
#[async_trait]
impl WasmBackend for RealWasmBackend {
    async fn execute_script(&self, _language: &str, script: &str) -> Result<String, ToolError> {
        let mut store = wasmtime::Store::new(&self.engine, ());
        store
            .set_fuel(self.fuel_budget)
            .map_err(|e| ToolError::InitializationFailed(e.to_string()))?;

        // No WASI imports — pure computational sandbox.
        let instance = wasmtime::Instance::new(&mut store, &self.module, &[])
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| ToolError::InitializationFailed("missing 'memory' export".into()))?;

        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|_| ToolError::InitializationFailed("missing 'alloc' export".into()))?;

        let eval = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, "eval")
            .map_err(|_| ToolError::InitializationFailed("missing 'eval' export".into()))?;

        let dealloc = instance
            .get_typed_func::<(i32, i32), ()>(&mut store, "dealloc")
            .map_err(|_| ToolError::InitializationFailed("missing 'dealloc' export".into()))?;

        let script_bytes = script.as_bytes();
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let len = script_bytes.len() as i32;

        let ptr = alloc
            .call(&mut store, len)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        #[allow(clippy::cast_sign_loss)]
        memory
            .data_mut(&mut store)
            .get_mut(ptr as usize..ptr as usize + script_bytes.len())
            .ok_or_else(|| {
                ToolError::ExecutionFailed("alloc returned out-of-bounds pointer".into())
            })?
            .copy_from_slice(script_bytes);

        let res_ptr = eval.call(&mut store, (ptr, len)).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("fuel") {
                ToolError::ExecutionFailed("fuel exhausted".into())
            } else {
                ToolError::ExecutionFailed(msg)
            }
        })?;

        let data = memory.data(&store);
        #[allow(clippy::cast_sign_loss)]
        let start = res_ptr as usize;
        let end = data[start..]
            .iter()
            .position(|&b| b == 0)
            .map_or(data.len(), |p| start + p);
        let result = String::from_utf8_lossy(&data[start..end]).into_owned();

        let _ = dealloc.call(&mut store, (ptr, len));

        Ok(result)
    }
}

// ── Executor ─────────────────────────────────────────────────────────────────

pub struct WasmExecutor {
    backend: Box<dyn WasmBackend>,
}

impl WasmExecutor {
    #[must_use]
    pub fn new(backend: Box<dyn WasmBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl ToolExecutor for WasmExecutor {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "code_execution",
            description: "Execute a JavaScript script inside a sandboxed WASM interpreter. Returns the evaluation result. No network or filesystem access.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["javascript"],
                        "description": "Script language. Only 'javascript' is supported."
                    },
                    "script": {
                        "type": "string",
                        "description": "JavaScript expression or program to evaluate."
                    }
                },
                "required": ["language", "script"]
            }),
        }
    }

    async fn execute(&self, input: &str) -> Result<String, ToolError> {
        let v: serde_json::Value =
            serde_json::from_str(input).map_err(|e| ToolError::MalformedInput(e.to_string()))?;

        let language = v["language"]
            .as_str()
            .ok_or_else(|| ToolError::MalformedInput("missing 'language' field".into()))?;
        let script = v["script"]
            .as_str()
            .ok_or_else(|| ToolError::MalformedInput("missing 'script' field".into()))?;

        if language != "javascript" {
            return Err(ToolError::NotPermitted(format!(
                "unsupported language: {language}"
            )));
        }

        self.backend.execute_script(language, script).await
    }
}
