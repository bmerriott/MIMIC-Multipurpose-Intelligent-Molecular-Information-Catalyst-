/**
 * License Key Generator Tool
 * 
 * This module provides utilities for generating license keys.
 * It should be used by the developer (you) to create keys for Patreon subscribers.
 * 
 * Usage (run from project root):
 *   cargo run --bin license-keygen -- --months 1 --type transferable
 * 
 * Or integrate into a web service for automated Patreon webhooks.
 */

use clap::{Arg, Command};
use hmac::{Hmac, Mac};
use sha2::Sha256;

// Base32 alphabet (omitting confusing characters: 0, O, 1, I, L)
const BASE32_ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";

/// License generation parameters
pub struct LicenseParams {
    pub license_type: LicenseType,
    pub months: u32,
    pub machine_id: Option<String>,
    pub subscriber_id: Option<String>,
    pub secret: String,
}

#[derive(Debug, Clone)]
pub enum LicenseType {
    MachineBound,
    Transferable,
}

impl LicenseType {
    fn as_char(&self) -> char {
        match self {
            LicenseType::MachineBound => 'M',
            LicenseType::Transferable => 'T',
        }
    }
    
    fn as_str(&self) -> &'static str {
        match self {
            LicenseType::MachineBound => "machine_bound",
            LicenseType::Transferable => "transferable",
        }
    }
}

impl std::str::FromStr for LicenseType {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "machine_bound" | "machine-bound" | "machine" | "m" => {
                Ok(LicenseType::MachineBound)
            }
            "transferable" | "transfer" | "t" => Ok(LicenseType::Transferable),
            _ => Err(format!("Unknown license type: {}", s)),
        }
    }
}

/// Generate a license key
pub fn generate_license_key(params: &LicenseParams) -> Result<String, String> {
    let version: u8 = 1;
    
    // Calculate expiry date
    let now = chrono::Utc::now();
    let expiry = now + chrono::Duration::days((params.months * 30 + 1) as i64); // +1 day grace
    
    // Encode expiry (seconds since 2024-01-01)
    let epoch = chrono::Utc
        .with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
        .single()
        .ok_or("Failed to create epoch date")?;
    let seconds = (expiry - epoch).num_seconds() as u64;
    
    let mut expiry_bytes = [0u8; 6];
    for i in 0..6 {
        expiry_bytes[5 - i] = ((seconds >> (i * 8)) & 0xFF) as u8;
    }
    let expiry_encoded = base32_encode(&expiry_bytes);
    
    // Encode machine ID
    let machine_encoded = params
        .machine_id
        .as_ref()
        .map(|m| clean_id(m, 8))
        .unwrap_or_else(|| "XXXXXXXX".to_string());
    
    // Encode subscriber ID
    let subscriber_encoded = params
        .subscriber_id
        .as_ref()
        .map(|s| clean_id(s, 8))
        .unwrap_or_else(|| "XXXXXXXX".to_string());
    
    // Build payload: VV T EEEEEEEEEE MMMMMMMM SSSSSSSS
    // V = version (2 chars)
    // T = type (1 char: M/T)
    // E = expiry (10 chars base32)
    // M = machine ID (8 chars)
    // S = subscriber ID (8 chars)
    let payload = format!(
        "{:02}{}{}{}{}",
        version,
        params.license_type.as_char(),
        expiry_encoded,
        machine_encoded,
        subscriber_encoded
    );
    
    // Generate HMAC-SHA256 signature
    let signature = generate_hmac(&payload, &params.secret);
    let signature_truncated = &signature[..12.min(signature.len())];
    
    // Combine payload + signature
    let combined = format!("{}{}", payload, signature_truncated);
    
    // Format as readable key
    let formatted = format_license_key(&combined);
    
    Ok(formatted)
}

/// Generate multiple license keys at once
pub fn generate_license_keys(
    params: &LicenseParams,
    count: u32,
) -> Result<Vec<String>, String> {
    (0..count)
        .map(|_| generate_license_key(params))
        .collect()
}

/// Clean and truncate an ID string
fn clean_id(id: &str, max_len: usize) -> String {
    let cleaned: String = id
        .to_uppercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    
    let truncated = &cleaned[..cleaned.len().min(max_len)];
    format!("{:X<width$}", truncated, width = max_len)
}

/// Encode bytes to base32 string
fn base32_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| BASE32_ALPHABET[b as usize % BASE32_ALPHABET.len()] as char)
        .collect()
}

/// Format key as MIMIC-XXXX-XXXX-XXXX-XXXX
fn format_license_key(data: &str) -> String {
    let groups: Vec<String> = data
        .chars()
        .collect::<Vec<_>>()
        .chunks(4)
        .map(|chunk| chunk.iter().collect())
        .collect();
    
    format!("MIMIC-{}", groups.join("-"))
}

/// Generate HMAC-SHA256 signature
fn generate_hmac(message: &str, secret: &str) -> String {
    type HmacSha256 = Hmac<Sha256>;
    
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    
    mac.update(message.as_bytes());
    let result = mac.finalize();
    let bytes = result.into_bytes();
    
    hex::encode(bytes)
}

/// Decode a license key (for verification)
pub fn decode_license_key(key: &str) -> Option<DecodedLicense> {
    let clean = key.trim().to_uppercase();
    
    if !clean.starts_with("MIMIC-") {
        return None;
    }
    
    let data: String = clean.chars().filter(|c| c.is_alphanumeric()).collect();
    
    if data.len() < 24 {
        return None;
    }
    
    // Skip "MIMIC" prefix
    let payload_start = 5;
    let payload = &data[payload_start..payload_start + 28];
    let signature = &data[payload_start + 28..data.len().min(payload_start + 40)];
    
    // Parse version
    let version: u8 = payload[0..2].parse().ok()?;
    
    // Parse type
    let license_type = match payload.chars().nth(2) {
        Some('M') => LicenseType::MachineBound,
        Some('T') => LicenseType::Transferable,
        _ => return None,
    };
    
    // Parse expiry
    let expiry_encoded = &payload[3..13];
    let expires_at = decode_expiry_date(expiry_encoded)?;
    
    // Parse machine ID
    let machine_encoded = &payload[13..21];
    let machine_id = if machine_encoded != "XXXXXXXX" {
        Some(machine_encoded.to_string())
    } else {
        None
    };
    
    // Parse subscriber ID
    let subscriber_encoded = &payload[21..29];
    let subscriber_id = if subscriber_encoded != "XXXXXXXX" {
        Some(subscriber_encoded.to_string())
    } else {
        None
    };
    
    Some(DecodedLicense {
        version,
        license_type,
        expires_at,
        machine_id,
        subscriber_id,
        signature: signature.to_string(),
        payload: payload.to_string(),
    })
}

/// Decoded license information
pub struct DecodedLicense {
    pub version: u8,
    pub license_type: LicenseType,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub machine_id: Option<String>,
    pub subscriber_id: Option<String>,
    pub signature: String,
    pub payload: String,
}

/// Decode expiry date from base32
fn decode_expiry_date(encoded: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if encoded.len() != 10 {
        return None;
    }
    
    let bytes = base32_decode(encoded)?;
    if bytes.len() < 6 {
        return None;
    }
    
    let mut seconds: u64 = 0;
    for i in 0..6 {
        seconds = (seconds << 8) | (bytes[i] as u64);
    }
    
    let epoch = chrono::Utc
        .with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
        .single()?;
    
    Some(epoch + chrono::Duration::seconds(seconds as i64))
}

/// Decode base32 string
fn base32_decode(encoded: &str) -> Option<Vec<u8>> {
    let mut result = Vec::new();
    
    for c in encoded.chars() {
        let c = c.to_ascii_uppercase();
        if let Some(pos) = BASE32_ALPHABET.iter().position(|&b| b as char == c) {
            result.push(pos as u8);
        } else {
            return None;
        }
    }
    
    Some(result)
}

/// Verify a license key
pub fn verify_license_key(
    key: &str,
    secret: &str,
    current_machine_id: Option<&str>,
) -> Result<VerificationResult, String> {
    let decoded = decode_license_key(key)
        .ok_or_else(|| "Invalid license key format".to_string())?;
    
    let now = chrono::Utc::now();
    let expired = now > decoded.expires_at;
    
    // Check machine binding
    let machine_mismatch = match (&decoded.license_type, &decoded.machine_id, current_machine_id) {
        (LicenseType::MachineBound, Some(bound_id), Some(current_id)) => {
            let bound_prefix = &bound_id[..bound_id.len().min(8)];
            let current_prefix = &current_id[..current_id.len().min(8)];
            bound_prefix != current_prefix
        }
        _ => false,
    };
    
    // Verify signature
    let signature_valid = {
        let expected = generate_hmac(&decoded.payload, secret);
        let expected_truncated = &expected[..12.min(expected.len())];
        constant_time_compare(&decoded.signature, expected_truncated)
    };
    
    let valid = signature_valid && !expired && !machine_mismatch;
    
    let message = if !signature_valid {
        "Invalid signature"
    } else if expired {
        "License expired"
    } else if machine_mismatch {
        "Machine mismatch"
    } else {
        "Valid"
    };
    
    Ok(VerificationResult {
        valid,
        expired,
        machine_mismatch,
        signature_valid,
        message: message.to_string(),
        decoded,
    })
}

/// Verification result
pub struct VerificationResult {
    pub valid: bool,
    pub expired: bool,
    pub machine_mismatch: bool,
    pub signature_valid: bool,
    pub message: String,
    pub decoded: DecodedLicense,
}

/// Constant-time string comparison
fn constant_time_compare(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut result = 0u8;
    for (a_byte, b_byte) in a.bytes().zip(b.bytes()) {
        result |= a_byte ^ b_byte;
    }
    
    result == 0
}

/// CLI entry point
pub fn main() {
    let matches = Command::new("Mimic AI License Key Generator")
        .version("1.0.0")
        .author("Mimic AI")
        .about("Generate license keys for Patreon subscribers")
        .arg(
            Arg::new("type")
                .short('t')
                .long("type")
                .value_name("TYPE")
                .help("License type: machine_bound or transferable")
                .default_value("transferable"),
        )
        .arg(
            Arg::new("months")
                .short('m')
                .long("months")
                .value_name("MONTHS")
                .help("Number of months the license is valid")
                .default_value("1"),
        )
        .arg(
            Arg::new("machine-id")
                .long("machine-id")
                .value_name("ID")
                .help("Machine ID to bind the license to (for machine_bound type)"),
        )
        .arg(
            Arg::new("subscriber")
                .short('s')
                .long("subscriber")
                .value_name("ID")
                .help("Subscriber ID or email (embedded in key)"),
        )
        .arg(
            Arg::new("secret")
                .long("secret")
                .value_name("SECRET")
                .help("HMAC secret key")
                .env("MIMIC_LICENSE_SECRET")
                .default_value("mimic-ai-production-secret"),
        )
        .arg(
            Arg::new("count")
                .short('c')
                .long("count")
                .value_name("COUNT")
                .help("Generate multiple keys")
                .default_value("1"),
        )
        .arg(
            Arg::new("verify")
                .short('v')
                .long("verify")
                .value_name("KEY")
                .help("Verify a license key instead of generating"),
        )
        .get_matches();
    
    // Verify mode
    if let Some(key) = matches.get_one::<String>("verify") {
        let secret = matches
            .get_one::<String>("secret")
            .cloned()
            .unwrap_or_default();
        let machine_id = matches.get_one::<String>("machine-id").map(|s| s.as_str());
        
        match verify_license_key(key, &secret, machine_id) {
            Ok(result) => {
                println!("Verification Result:");
                println!("  Valid: {}", result.valid);
                println!("  Message: {}", result.message);
                println!("  Type: {:?}", result.decoded.license_type);
                println!("  Expires: {}", result.decoded.expires_at.format("%Y-%m-%d %H:%M:%S UTC"));
                if let Some(ref mid) = result.decoded.machine_id {
                    println!("  Machine ID: {}", mid);
                }
                if let Some(ref sid) = result.decoded.subscriber_id {
                    println!("  Subscriber ID: {}", sid);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }
    
    // Generation mode
    let license_type = matches
        .get_one::<String>("type")
        .unwrap()
        .parse::<LicenseType>()
        .expect("Invalid license type");
    
    let months = matches
        .get_one::<String>("months")
        .unwrap()
        .parse::<u32>()
        .expect("Invalid months value");
    
    let machine_id = matches.get_one::<String>("machine-id").cloned();
    let subscriber_id = matches.get_one::<String>("subscriber").cloned();
    let secret = matches
        .get_one::<String>("secret")
        .cloned()
        .unwrap_or_default();
    let count = matches
        .get_one::<String>("count")
        .unwrap()
        .parse::<u32>()
        .expect("Invalid count value");
    
    let params = LicenseParams {
        license_type,
        months,
        machine_id,
        subscriber_id,
        secret,
    };
    
    println!("Generating {} license key(s)...", count);
    println!("  Type: {:?}", params.license_type);
    println!("  Duration: {} month(s)", params.months);
    println!("  Secret: {}...", &params.secret[..params.secret.len().min(8)]);
    println!();
    
    for i in 1..=count {
        match generate_license_key(&params) {
            Ok(key) => {
                println!("Key {}: {}", i, key);
                
                // Also decode and show info
                if let Some(decoded) = decode_license_key(&key) {
                    println!("  Expires: {}", decoded.expires_at.format("%Y-%m-%d"));
                    if let Some(ref sid) = decoded.subscriber_id {
                        println!("  Subscriber: {}", sid);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error generating key {}: {}", i, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_verify() {
        let params = LicenseParams {
            license_type: LicenseType::Transferable,
            months: 1,
            machine_id: None,
            subscriber_id: Some("test123".to_string()),
            secret: "test-secret".to_string(),
        };
        
        let key = generate_license_key(&params).expect("Failed to generate key");
        assert!(key.starts_with("MIMIC-"));
        
        // Verify
        let result = verify_license_key(&key, &params.secret, None)
            .expect("Failed to verify");
        
        assert!(result.valid);
        assert!(!result.expired);
        assert_eq!(result.decoded.subscriber_id, Some("TEST123 ".to_string()));
    }

    #[test]
    fn test_machine_bound_mismatch() {
        let params = LicenseParams {
            license_type: LicenseType::MachineBound,
            months: 1,
            machine_id: Some("MACHINE123".to_string()),
            subscriber_id: None,
            secret: "test-secret".to_string(),
        };
        
        let key = generate_license_key(&params).expect("Failed to generate key");
        
        // Verify with wrong machine ID
        let result = verify_license_key(&key, &params.secret, Some("WRONG123"))
            .expect("Failed to verify");
        
        assert!(!result.valid);
        assert!(result.machine_mismatch);
    }
}
