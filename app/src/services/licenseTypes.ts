/**
 * License Types for Mimic AI Desktop Assistant
 * Defines all TypeScript types for the licensing/subscription system
 */

/** License status types */
export type LicenseStatus = 
  | 'trial_active'      // Within 7-day trial period
  | 'trial_expired'     // Trial has expired, no license
  | 'licensed_active'   // Has valid license, not expired
  | 'licensed_expired'  // License exists but has expired
  | 'invalid';          // License key is invalid/corrupted

/** License key type */
export type LicenseType = 
  | 'machine_bound'     // Tied to specific machine ID
  | 'transferable';     // Can be used on any machine (manual management)

/** License data stored locally */
export interface LicenseData {
  /** The license key string */
  key: string;
  /** Type of license */
  type: LicenseType;
  /** When the license was first activated */
  activatedAt: string;
  /** When the license expires */
  expiresAt: string;
  /** Machine ID this license is bound to (if machine_bound) */
  machineId?: string;
  /** Patreon subscriber ID or email */
  subscriberId?: string;
  /** Signature for verification */
  signature: string;
}

/** Trial data stored locally (obfuscated) */
export interface TrialData {
  /** When the trial started */
  startedAt: string;
  /** Whether trial has been used before */
  hasStarted: boolean;
  /** Machine ID where trial was started */
  machineId: string;
  /** Encrypted/obfuscated storage to prevent tampering */
  checksum: string;
}

/** License verification result */
export interface LicenseVerificationResult {
  /** Whether the license is valid */
  valid: boolean;
  /** Current status */
  status: LicenseStatus;
  /** Days remaining (for trial or license) */
  daysRemaining: number;
  /** Human-readable message */
  message: string;
  /** License data (if valid) */
  license?: LicenseData;
  /** Trial data (if applicable) */
  trial?: TrialData;
}

/** Decoded license key data */
export interface DecodedLicenseKey {
  /** Version of the license key format */
  version: number;
  /** License type */
  type: LicenseType;
  /** Expiry timestamp */
  expiresAt: string;
  /** Optional machine ID binding */
  machineId?: string;
  /** Optional subscriber identifier */
  subscriberId?: string;
  /** HMAC signature */
  signature: string;
  /** Raw payload that was signed */
  payload: string;
}

/** Machine fingerprint data */
export interface MachineFingerprint {
  /** Unique machine ID hash */
  id: string;
  /** Components used to generate ID (hashed) */
  components: {
    cpu: string;
    motherboard: string;
    os: string;
    hostname: string;
  };
  /** When the fingerprint was generated */
  generatedAt: string;
}

/** Patreon subscription tiers */
export type SubscriptionTier = 
  | 'none'
  | 'supporter'      // $5/month - basic license
  | 'patron';        // Higher tier - could have additional features

/** Subscription info from Patreon */
export interface SubscriptionInfo {
  tier: SubscriptionTier;
  status: 'active' | 'cancelled' | 'expired' | 'pending';
  startedAt: string;
  expiresAt?: string;
  pledgeAmount: number;
}

/** License manager configuration */
export interface LicenseConfig {
  /** Trial duration in days */
  trialDays: number;
  /** Storage key for localStorage */
  storageKey: string;
  /** Storage key for trial data */
  trialStorageKey: string;
  /** HMAC secret (should match Rust backend) */
  hmacSecret: string;
  /** Patreon page URL */
  patreonUrl: string;
  /** GitHub releases URL */
  githubReleasesUrl: string;
}

/** Default license configuration */
export const DEFAULT_LICENSE_CONFIG: LicenseConfig = {
  trialDays: 7,
  storageKey: 'mimic_license_v1',
  trialStorageKey: 'mimic_trial_v1',
  hmacSecret: (typeof import.meta !== 'undefined' && import.meta.env?.VITE_LICENSE_SECRET) || 'mimic-ai-default-secret-change-in-production',
  patreonUrl: 'https://www.patreon.com/mimicai',
  githubReleasesUrl: 'https://github.com/yourusername/mimic-ai/releases',
};
