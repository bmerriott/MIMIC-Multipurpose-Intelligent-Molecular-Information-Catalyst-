/**
 * License Manager for Mimic AI Desktop Assistant
 * 
 * Simple version for test build - no registry access
 */

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// License verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseVerificationResult {
    pub valid: bool,
    pub expired: bool,
    pub machine_mismatch: bool,
    pub message: String,
}

/// Get machine ID (simple non-intrusive version)
#[tauri::command]
pub async fn get_machine_id() -> Result<String, String> {
    generate_machine_id().await
}

/// Generate a unique machine ID using hostname and random data
async fn generate_machine_id() -> Result<String, String> {
    // Get hostname (non-sensitive, no registry access)
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string());
    
    // Get OS info via standard Rust
    let os_info = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    
    // Combine and hash
    let combined = format!("{}|{}|{}", hostname, os_info, arch);
    let hash = calculate_hash(&combined);
    
    // Format as readable string (32 chars, hex)
    let machine_id = format!("{:016X}{:016X}", hash, hash.wrapping_mul(0x9E3779B97F4A7C15));
    
    Ok(machine_id)
}

/// Calculate hash of a string
fn calculate_hash(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Verify license key (placeholder for test build)
#[tauri::command]
pub async fn verify_license_key(key: String, _machine_id: Option<String>) -> Result<LicenseVerificationResult, String> {
    // Test build - all keys valid for testing
    Ok(LicenseVerificationResult {
        valid: true,
        expired: false,
        machine_mismatch: false,
        message: format!("Test build - key '{}' accepted", key),
    })
}

/// Generate license key (for developer use)
#[allow(dead_code)]
pub fn generate_license_key(_subscriber_id: &str, _months: u32) -> Result<String, String> {
    // Placeholder - in production this would generate signed keys
    Ok("MIMIC-TEST-0000-0000-0000".to_string())
}
