//! NATS JetStream topology and NKey provisioning for the H2AI Control Plane.
//!
//! - [`subjects`] — subject name constants and builders for the H2AI event bus
//! - [`nkey`] — ephemeral scoped NKey generation for edge agent containers

pub mod nkey;
pub mod subjects;
