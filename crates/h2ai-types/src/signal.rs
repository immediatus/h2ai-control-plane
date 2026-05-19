use crate::identity::{TaskId, TenantId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeSignal {
    pub task_id: TaskId,
    pub tenant_id: TenantId,
    #[serde(
        serialize_with = "serialize_signal_payload",
        deserialize_with = "deserialize_signal_payload"
    )]
    pub payload: SignalPayload,
    /// Absolute Unix-ms deadline. Engine discards signal if `now_ms() > timeout_at_ms`.
    pub timeout_at_ms: u64,
    pub issued_at_ms: u64,
}

/// External contract — crosses the NATS boundary.
/// `Unknown` catches future variants at serde deserialization; maps to `ResumeAction::Ignore`.
#[derive(Debug, Clone)]
pub enum SignalPayload {
    WaveContinue(WaveContinueSignal),
    Approve(ApproveSignal),
    /// Catches unknown future variants at runtime — prevents serde errors on old binaries.
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveContinueSignal {
    pub grounding: Option<String>,
    pub mandate_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproveSignal {
    pub approved: bool,
    pub reviewer_note: Option<String>,
    pub operator_id: String,
}

/// Custom serializer for SignalPayload using adjacently-tagged format.
fn serialize_signal_payload<S>(payload: &SignalPayload, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use serde::ser::SerializeMap;

    match payload {
        SignalPayload::WaveContinue(signal) => {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("kind", "WaveContinue")?;
            map.serialize_entry("data", signal)?;
            map.end()
        }
        SignalPayload::Approve(signal) => {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("kind", "Approve")?;
            map.serialize_entry("data", signal)?;
            map.end()
        }
        SignalPayload::Unknown => {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("kind", "Unknown")?;
            map.serialize_entry("data", &serde_json::json!({}))?;
            map.end()
        }
    }
}

/// Custom deserializer for SignalPayload that catches unknown variants.
fn deserialize_signal_payload<'de, D>(deserializer: D) -> Result<SignalPayload, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(Deserialize)]
    struct Envelope {
        kind: String,
        data: serde_json::Value,
    }

    let env = Envelope::deserialize(deserializer)?;

    match env.kind.as_str() {
        "WaveContinue" => {
            let signal: WaveContinueSignal =
                serde_json::from_value(env.data).map_err(D::Error::custom)?;
            Ok(SignalPayload::WaveContinue(signal))
        }
        "Approve" => {
            let signal: ApproveSignal =
                serde_json::from_value(env.data).map_err(D::Error::custom)?;
            Ok(SignalPayload::Approve(signal))
        }
        _ => Ok(SignalPayload::Unknown),
    }
}
