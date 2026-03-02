/**
 * Patreon License Key Generation and Validation
 * 
 * This module handles:
 * - License key format: MIMIC-XXXX-XXXX-XXXX-XXXX
 * - Cryptographic signing with HMAC-SHA256
 * - Offline verification (no server required after initial key generation)
 * 
 * NOTE: The actual key generation should be done by YOU (the developer) using
 * the generateLicenseKey() function or a separate tool. Users receive pre-made keys.
 * 
 * SECURITY WARNING: The HMAC secret should be kept private and never exposed
 * in the client-side code. In production, use environment variables and build-time
 * injection, or better yet, verify signatures in the Rust backend only.
 */

import type { LicenseType, DecodedLicenseKey, LicenseData } from './licenseTypes';

// License key format: MIMIC-XXXX-XXXX-XXXX-XXXX (24 characters total, 20 without dashes)
const LICENSE_KEY_REGEX = /^MIMIC-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/;

// Base32 alphabet for encoding (omitting confusing characters: 0, O, 1, I, L)
const BASE32_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';

/**
 * Generate a cryptographically secure random string
 */


/**
 * Encode bytes to Base32 string
 */
function base32Encode(buffer: Uint8Array): string {
  let result = '';
  for (let i = 0; i < buffer.length; i++) {
    result += BASE32_ALPHABET[buffer[i] % BASE32_ALPHABET.length];
  }
  return result;
}

/**
 * Decode Base32 string to bytes
 */
function base32Decode(str: string): Uint8Array {
  const result = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) {
    const char = str[i].toUpperCase();
    const index = BASE32_ALPHABET.indexOf(char);
    result[i] = index >= 0 ? index : 0;
  }
  return result;
}

/**
 * Convert string to Uint8Array
 */
function stringToBytes(str: string): Uint8Array {
  return new TextEncoder().encode(str);
}

/**
 * Convert Uint8Array to hex string
 */
function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Generate HMAC-SHA256 signature
 * In production, this should be done in the Rust backend to keep the secret safe
 */
async function generateHmac(message: string, secret: string): Promise<string> {
  const keyData = stringToBytes(secret);
  const messageData = stringToBytes(message);
  
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    keyData.buffer as ArrayBuffer,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  
  const signature = await crypto.subtle.sign('HMAC', cryptoKey, messageData.buffer as ArrayBuffer);
  return bytesToHex(new Uint8Array(signature));
}

/**
 * Verify HMAC-SHA256 signature
 */
async function verifyHmac(message: string, signature: string, secret: string): Promise<boolean> {
  const expectedSignature = await generateHmac(message, secret);
  // Constant-time comparison to prevent timing attacks
  if (signature.length !== expectedSignature.length) {
    return false;
  }
  let result = 0;
  for (let i = 0; i < signature.length; i++) {
    result |= signature.charCodeAt(i) ^ expectedSignature.charCodeAt(i);
  }
  return result === 0;
}

/**
 * Pack license data into a compact format for the key
 */
function packLicenseData(
  version: number,
  type: LicenseType,
  expiresAt: Date,
  machineId?: string,
  subscriberId?: string
): string {
  // Format: VV_T_EEEE..._MMMM..._SSSS...
  // V = version (2 chars)
  // T = type (1 char: M=machine_bound, T=transferable)
  // E = expiry timestamp base32 encoded (8 chars = ~48 bits, enough until year 2100+)
  // M = machine ID hash prefix (optional, 8 chars)
  // S = subscriber ID hash (optional, 8 chars)
  
  const typeChar = type === 'machine_bound' ? 'M' : 'T';
  
  // Encode expiry as 6-byte timestamp (seconds since 2024)
  const EPOCH_START = new Date('2024-01-01T00:00:00Z').getTime();
  const expirySeconds = Math.floor((expiresAt.getTime() - EPOCH_START) / 1000);
  const expiryBytes = new Uint8Array(6);
  for (let i = 0; i < 6; i++) {
    expiryBytes[5 - i] = (expirySeconds >> (i * 8)) & 0xFF;
  }
  const expiryEncoded = base32Encode(expiryBytes).padStart(10, 'A');
  
  // Machine ID prefix (first 8 chars of hash, or 'XXXXXXXX' if not bound)
  const machineEncoded = machineId 
    ? machineId.replace(/[^A-Z0-9]/g, '').slice(0, 8).padEnd(8, 'X')
    : 'XXXXXXXX';
  
  // Subscriber ID hash (simplified, first 8 chars)
  const subscriberEncoded = subscriberId
    ? subscriberId.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8).padEnd(8, 'X')
    : 'XXXXXXXX';
  
  return `${version.toString().padStart(2, '0')}${typeChar}${expiryEncoded}${machineEncoded}${subscriberEncoded}`;
}

/**
 * Unpack license data from packed format
 */
function unpackLicenseData(packed: string): {
  version: number;
  type: LicenseType;
  expiresAt: Date;
  machineId?: string;
  subscriberId?: string;
} {
  const version = parseInt(packed.slice(0, 2), 10);
  const typeChar = packed[2];
  const type: LicenseType = typeChar === 'M' ? 'machine_bound' : 'transferable';
  
  // Decode expiry
  const expiryEncoded = packed.slice(3, 13);
  const expiryBytes = base32Decode(expiryEncoded);
  let expirySeconds = 0;
  for (let i = 0; i < 6; i++) {
    expirySeconds = (expirySeconds << 8) | expiryBytes[i];
  }
  const EPOCH_START = new Date('2024-01-01T00:00:00Z').getTime();
  const expiresAt = new Date(EPOCH_START + expirySeconds * 1000);
  
  // Decode machine ID
  const machineEncoded = packed.slice(13, 21);
  const machineId = machineEncoded !== 'XXXXXXXX' ? machineEncoded : undefined;
  
  // Decode subscriber ID
  const subscriberEncoded = packed.slice(21, 29);
  const subscriberId = subscriberEncoded !== 'XXXXXXXX' ? subscriberEncoded : undefined;
  
  return { version, type, expiresAt, machineId, subscriberId };
}

/**
 * Format a license key in readable format: MIMIC-XXXX-XXXX-XXXX-XXXX
 */
function formatLicenseKey(data: string, signature: string): string {
  // Combine data and truncated signature
  const combined = data + signature.slice(0, 12);
  
  // Format as groups of 4
  const groups = [];
  for (let i = 0; i < combined.length; i += 4) {
    groups.push(combined.slice(i, i + 4));
  }
  
  return 'MIMIC-' + groups.join('-');
}

/**
 * Parse a formatted license key
 */
function parseLicenseKey(key: string): { data: string; signature: string } | null {
  if (!LICENSE_KEY_REGEX.test(key)) {
    return null;
  }
  
  // Remove prefix and dashes
  const clean = key.replace(/^MIMIC-/, '').replace(/-/g, '');
  
  // Split into data (28 chars) and signature (12 chars)
  const data = clean.slice(0, 28);
  const signature = clean.slice(28);
  
  return { data, signature };
}

/**
 * Generate a new license key
 * 
 * THIS SHOULD ONLY BE USED BY THE DEVELOPER to generate keys for Patreon subscribers.
 * Never expose this function or the HMAC secret to end users.
 * 
 * @param type - Type of license (machine_bound or transferable)
 * @param months - Number of months the license is valid for
 * @param machineId - Optional machine ID to bind to
 * @param subscriberId - Optional subscriber identifier from Patreon
 * @param secret - HMAC secret key (keep this private!)
 * @returns Formatted license key
 */
export async function generateLicenseKey(
  type: LicenseType,
  months: number,
  machineId?: string,
  subscriberId?: string,
  secret: string = 'mimic-ai-production-secret'
): Promise<string> {
  // Calculate expiry date
  const expiresAt = new Date();
  expiresAt.setMonth(expiresAt.getMonth() + months);
  expiresAt.setDate(expiresAt.getDate() + 1); // Add 1 day grace period
  
  // Pack license data
  const version = 1;
  const packedData = packLicenseData(version, type, expiresAt, machineId, subscriberId);
  
  // Generate signature
  const signature = await generateHmac(packedData, secret);
  
  // Format key
  return formatLicenseKey(packedData, signature);
}

/**
 * Decode and verify a license key
 * 
 * @param key - The license key to decode
 * @returns Decoded license data or null if invalid format
 */
export function decodeLicenseKey(key: string): DecodedLicenseKey | null {
  const parsed = parseLicenseKey(key);
  if (!parsed) {
    return null;
  }
  
  const unpacked = unpackLicenseData(parsed.data);
  
  return {
    version: unpacked.version,
    type: unpacked.type,
    expiresAt: unpacked.expiresAt.toISOString(),
    machineId: unpacked.machineId,
    subscriberId: unpacked.subscriberId,
    signature: parsed.signature,
    payload: parsed.data,
  };
}

/**
 * Verify a license key signature
 * 
 * @param key - The license key to verify
 * @param machineId - Current machine ID (for machine-bound licenses)
 * @param secret - HMAC secret key
 * @returns Verification result
 */
export async function verifyLicenseKey(
  key: string,
  machineId?: string,
  secret: string = 'mimic-ai-production-secret'
): Promise<{
  valid: boolean;
  expired: boolean;
  machineMismatch: boolean;
  decoded: DecodedLicenseKey | null;
}> {
  const decoded = decodeLicenseKey(key);
  
  if (!decoded) {
    return { valid: false, expired: false, machineMismatch: false, decoded: null };
  }
  
  // Check expiry
  const now = new Date();
  const expiry = new Date(decoded.expiresAt);
  const expired = now > expiry;
  
  // Check machine binding
  let machineMismatch = false;
  if (decoded.type === 'machine_bound' && decoded.machineId && machineId) {
    // Allow partial match (first 8 chars)
    const boundPrefix = decoded.machineId.slice(0, 8);
    const currentPrefix = machineId.replace(/[^A-Z0-9]/g, '').slice(0, 8);
    machineMismatch = boundPrefix !== currentPrefix;
  }
  
  // Verify signature
  const signatureValid = await verifyHmac(decoded.payload, decoded.signature, secret);
  
  return {
    valid: signatureValid && !expired && !machineMismatch,
    expired,
    machineMismatch,
    decoded,
  };
}

/**
 * Generate multiple license keys (for batch generation)
 * 
 * @param count - Number of keys to generate
 * @param type - License type
 * @param months - Validity period in months
 * @param secret - HMAC secret
 * @returns Array of license keys
 */
export async function generateLicenseKeys(
  count: number,
  type: LicenseType,
  months: number,
  secret: string = 'mimic-ai-production-secret'
): Promise<string[]> {
  const keys: string[] = [];
  for (let i = 0; i < count; i++) {
    const key = await generateLicenseKey(type, months, undefined, undefined, secret);
    keys.push(key);
  }
  return keys;
}

/**
 * Create license data object from decoded key
 */
export function createLicenseData(key: string, decoded: DecodedLicenseKey): LicenseData {
  return {
    key,
    type: decoded.type,
    activatedAt: new Date().toISOString(),
    expiresAt: decoded.expiresAt,
    machineId: decoded.machineId,
    subscriberId: decoded.subscriberId,
    signature: decoded.signature,
  };
}

/**
 * Get days until license expires
 */
export function getDaysUntilExpiry(expiresAt: string | Date): number {
  const expiry = typeof expiresAt === 'string' ? new Date(expiresAt) : expiresAt;
  const now = new Date();
  const diff = expiry.getTime() - now.getTime();
  return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

/**
 * Check if license is expired
 */
export function isLicenseExpired(expiresAt: string | Date): boolean {
  const expiry = typeof expiresAt === 'string' ? new Date(expiresAt) : expiresAt;
  return new Date() > expiry;
}
