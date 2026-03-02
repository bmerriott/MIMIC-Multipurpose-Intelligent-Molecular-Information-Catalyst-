/**
 * License Manager for Mimic AI Desktop Assistant
 * 
 * Features:
 * - 7-day free trial from first launch
 * - Offline license verification using HMAC-signed keys
 * - Local storage of trial and license data (encrypted/obfuscated)
 * - No authentication server required after initial download
 * 
 * Usage:
 * 1. App starts -> checkLicenseStatus()
 * 2. If trial_active -> allow full access
 * 3. If trial_expired -> show license prompt
 * 4. User enters Patreon license key -> activateLicense(key)
 * 5. If licensed_active -> allow full access until expiry
 */

import { invoke } from '@tauri-apps/api/tauri';
import type {
  LicenseStatus,
  LicenseData,
  TrialData,
  LicenseVerificationResult,
  LicenseConfig,
} from './licenseTypes';
import { DEFAULT_LICENSE_CONFIG } from './licenseTypes';
import {
  decodeLicenseKey,
  verifyLicenseKey,
  createLicenseData,
  getDaysUntilExpiry,
  isLicenseExpired,
} from './patreonLicense';

// Storage keys (obfuscated names to deter casual tampering)
const STORAGE_KEYS = {
  trial: '_m_trial_data_v1',
  license: '_m_license_data_v1',
  machine: '_m_machine_id_v1',
  checksum: '_m_checksum_v1',
};

class LicenseManager {
  private config: LicenseConfig;
  private machineId: string | null = null;
  private cachedLicense: LicenseData | null = null;
  private cachedTrial: TrialData | null = null;
  private initialized: boolean = false;

  constructor(config: Partial<LicenseConfig> = {}) {
    this.config = { ...DEFAULT_LICENSE_CONFIG, ...config };
  }

  /**
   * Initialize the license manager
   * - Gets or generates machine ID
   * - Loads cached license/trial data
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Get machine ID from Rust backend
      this.machineId = await this.getMachineId();
      
      // Load cached data
      this.cachedLicense = await this.loadLicenseData();
      this.cachedTrial = await this.loadTrialData();
      
      this.initialized = true;
    } catch (error) {
      console.error('[LicenseManager] Initialization failed:', error);
      // Still mark as initialized to prevent repeated failures
      this.initialized = true;
    }
  }

  /**
   * Get or generate machine ID
   * Uses Rust backend to get hardware fingerprint
   */
  private async getMachineId(): Promise<string | null> {
    try {
      // Check if we have a cached machine ID
      const cached = localStorage.getItem(STORAGE_KEYS.machine);
      if (cached) {
        return cached;
      }

      // Get from Rust backend
      const machineId = await invoke<string>('get_machine_id');
      
      // Cache it
      localStorage.setItem(STORAGE_KEYS.machine, machineId);
      
      return machineId;
    } catch (error) {
      console.error('[LicenseManager] Failed to get machine ID:', error);
      // Fallback: generate a pseudo-machine ID from browser fingerprint
      return this.generateFallbackMachineId();
    }
  }

  /**
   * Generate a fallback machine ID when Tauri is not available
   * This is less secure but allows the app to work in browser mode
   */
  private generateFallbackMachineId(): string {
    // Combine various browser/machine identifiers
    const components = [
      navigator.userAgent,
      navigator.language,
      screen.width + 'x' + screen.height,
      screen.colorDepth.toString(),
      new Date().getTimezoneOffset().toString(),
      navigator.hardwareConcurrency?.toString() || 'unknown',
    ];
    
    const combined = components.join('|');
    
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    
    const machineId = Math.abs(hash).toString(16).toUpperCase().padStart(16, '0');
    localStorage.setItem(STORAGE_KEYS.machine, machineId);
    return machineId;
  }

  /**
   * Check the current license status
   * This is the main entry point for license checking
   */
  async checkLicenseStatus(): Promise<LicenseVerificationResult> {
    await this.initialize();

    // 1. Check if we have a valid license
    if (this.cachedLicense) {
      const daysRemaining = getDaysUntilExpiry(this.cachedLicense.expiresAt);
      const expired = isLicenseExpired(this.cachedLicense.expiresAt);
      
      if (!expired) {
        return {
          valid: true,
          status: 'licensed_active',
          daysRemaining,
          message: `License valid. Expires in ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''}.`,
          license: this.cachedLicense,
        };
      } else {
        return {
          valid: false,
          status: 'licensed_expired',
          daysRemaining: 0,
          message: 'Your license has expired. Please renew on Patreon.',
          license: this.cachedLicense,
        };
      }
    }

    // 2. Check trial status
    const trialStatus = await this.checkTrialStatus();
    if (trialStatus.status === 'trial_active') {
      return trialStatus;
    }

    // 3. Trial expired, no license
    return {
      valid: false,
      status: 'trial_expired',
      daysRemaining: 0,
      message: 'Your 7-day trial has expired. Please subscribe on Patreon to continue using Mimic AI.',
      trial: this.cachedTrial || undefined,
    };
  }

  /**
   * Check trial status
   */
  private async checkTrialStatus(): Promise<LicenseVerificationResult> {
    // Check if trial has started
    if (!this.cachedTrial) {
      // Start new trial
      await this.startTrial();
      return {
        valid: true,
        status: 'trial_active',
        daysRemaining: this.config.trialDays,
        message: `Welcome to Mimic AI! Your ${this.config.trialDays}-day free trial has started.`,
        trial: this.cachedTrial || undefined,
      };
    }

    // Verify trial integrity
    const isValidTrial = await this.verifyTrialIntegrity();
    if (!isValidTrial) {
      // Trial data was tampered with
      console.warn('[LicenseManager] Trial data integrity check failed');
      return {
        valid: false,
        status: 'trial_expired',
        daysRemaining: 0,
        message: 'Trial data is invalid. Please contact support.',
        trial: this.cachedTrial,
      };
    }

    // Calculate days remaining
    const startDate = new Date(this.cachedTrial.startedAt);
    const now = new Date();
    const diffTime = now.getTime() - startDate.getTime();
    const daysElapsed = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    const daysRemaining = Math.max(0, this.config.trialDays - daysElapsed);

    if (daysRemaining > 0) {
      return {
        valid: true,
        status: 'trial_active',
        daysRemaining,
        message: `Trial active. ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} remaining.`,
        trial: this.cachedTrial,
      };
    } else {
      return {
        valid: false,
        status: 'trial_expired',
        daysRemaining: 0,
        message: 'Your trial has expired.',
        trial: this.cachedTrial,
      };
    }
  }

  /**
   * Start a new trial
   */
  private async startTrial(): Promise<void> {
    const machineId = this.machineId || 'unknown';
    const startedAt = new Date().toISOString();
    
    // Create checksum to prevent tampering
    const checksum = await this.generateTrialChecksum(machineId, startedAt);
    
    const trialData: TrialData = {
      startedAt,
      hasStarted: true,
      machineId,
      checksum,
    };

    // Obfuscate and store
    const obfuscated = this.obfuscateData(JSON.stringify(trialData));
    localStorage.setItem(STORAGE_KEYS.trial, obfuscated);
    
    this.cachedTrial = trialData;
  }

  /**
   * Load trial data from storage
   */
  private async loadTrialData(): Promise<TrialData | null> {
    const stored = localStorage.getItem(STORAGE_KEYS.trial);
    if (!stored) return null;

    try {
      const deobfuscated = this.deobfuscateData(stored);
      return JSON.parse(deobfuscated) as TrialData;
    } catch (error) {
      console.error('[LicenseManager] Failed to load trial data:', error);
      return null;
    }
  }

  /**
   * Load license data from storage
   */
  private async loadLicenseData(): Promise<LicenseData | null> {
    const stored = localStorage.getItem(STORAGE_KEYS.license);
    if (!stored) return null;

    try {
      const deobfuscated = this.deobfuscateData(stored);
      return JSON.parse(deobfuscated) as LicenseData;
    } catch (error) {
      console.error('[LicenseManager] Failed to load license data:', error);
      return null;
    }
  }

  /**
   * Save license data to storage
   */
  private async saveLicenseData(license: LicenseData): Promise<void> {
    const obfuscated = this.obfuscateData(JSON.stringify(license));
    localStorage.setItem(STORAGE_KEYS.license, obfuscated);
    this.cachedLicense = license;
  }

  /**
   * Simple obfuscation to deter casual tampering
   * Note: This is not encryption, just makes it harder to edit by hand
   */
  private obfuscateData(data: string): string {
    // XOR with a rotating key based on storage key
    const key = 'mimic_ai_license_system_v1';
    let result = '';
    for (let i = 0; i < data.length; i++) {
      const charCode = data.charCodeAt(i) ^ key.charCodeAt(i % key.length);
      result += String.fromCharCode(charCode);
    }
    // Base64 encode
    return btoa(result);
  }

  /**
   * Deobfuscate data
   */
  private deobfuscateData(obfuscated: string): string {
    try {
      // Base64 decode
      const data = atob(obfuscated);
      
      // XOR with same key
      const key = 'mimic_ai_license_system_v1';
      let result = '';
      for (let i = 0; i < data.length; i++) {
        const charCode = data.charCodeAt(i) ^ key.charCodeAt(i % key.length);
        result += String.fromCharCode(charCode);
      }
      return result;
    } catch (error) {
      throw new Error('Failed to deobfuscate data');
    }
  }

  /**
   * Generate checksum for trial data
   */
  private async generateTrialChecksum(machineId: string, startedAt: string): Promise<string> {
    // Simple hash of machine ID + start date + secret
    const data = `${machineId}|${startedAt}|${this.config.hmacSecret}`;
    
    // Use SubtleCrypto if available
    if (typeof crypto !== 'undefined' && crypto.subtle) {
      const encoder = new TextEncoder();
      const dataBuffer = encoder.encode(data);
      const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
    }
    
    // Fallback simple hash
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).toUpperCase().padStart(16, '0');
  }

  /**
   * Verify trial data hasn't been tampered with
   */
  private async verifyTrialIntegrity(): Promise<boolean> {
    if (!this.cachedTrial) return false;
    
    const expectedChecksum = await this.generateTrialChecksum(
      this.cachedTrial.machineId,
      this.cachedTrial.startedAt
    );
    
    return this.cachedTrial.checksum === expectedChecksum;
  }

  /**
   * Activate a license key
   * 
   * @param key - The license key from Patreon
   * @returns Activation result
   */
  async activateLicense(key: string): Promise<{
    success: boolean;
    message: string;
    status?: LicenseStatus;
  }> {
    await this.initialize();

    // Validate key format
    const decoded = decodeLicenseKey(key);
    if (!decoded) {
      return {
        success: false,
        message: 'Invalid license key format. Expected: MIMIC-XXXX-XXXX-XXXX-XXXX',
      };
    }

    // Verify signature (try using Rust backend first, then fallback)
    let verification;
    try {
      // Try Rust backend verification (more secure)
      const result = await invoke<{ valid: boolean; expired: boolean; machine_mismatch: boolean }>(
        'verify_license_key',
        { key, machineId: this.machineId }
      );
      verification = {
        valid: result.valid,
        expired: result.expired,
        machineMismatch: result.machine_mismatch,
        decoded,
      };
    } catch (error) {
      // Fallback to TypeScript verification
      console.warn('[LicenseManager] Rust verification failed, using fallback:', error);
      verification = await verifyLicenseKey(key, this.machineId || undefined, this.config.hmacSecret);
    }

    if (verification.machineMismatch) {
      return {
        success: false,
        message: 'This license key is bound to a different machine. Please contact support.',
      };
    }

    if (verification.expired) {
      return {
        success: false,
        message: 'This license key has expired. Please renew on Patreon.',
      };
    }

    if (!verification.valid) {
      return {
        success: false,
        message: 'Invalid license key. Please check the key and try again.',
      };
    }

    // Create and save license data
    const licenseData = createLicenseData(key, decoded);
    await this.saveLicenseData(licenseData);

    const daysRemaining = getDaysUntilExpiry(licenseData.expiresAt);
    
    return {
      success: true,
      status: 'licensed_active',
      message: `License activated successfully! Valid for ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''}.`,
    };
  }

  /**
   * Deactivate current license (for logout/machine transfer)
   */
  async deactivateLicense(): Promise<void> {
    localStorage.removeItem(STORAGE_KEYS.license);
    this.cachedLicense = null;
  }

  /**
   * Get current license info for display
   */
  async getLicenseInfo(): Promise<{
    hasLicense: boolean;
    isTrial: boolean;
    daysRemaining: number;
    expiryDate?: Date;
    type?: string;
  }> {
    const status = await this.checkLicenseStatus();
    
    return {
      hasLicense: status.status === 'licensed_active' || status.status === 'licensed_expired',
      isTrial: status.status === 'trial_active',
      daysRemaining: status.daysRemaining,
      expiryDate: status.license?.expiresAt 
        ? new Date(status.license.expiresAt) 
        : status.trial?.startedAt 
          ? new Date(new Date(status.trial.startedAt).getTime() + this.config.trialDays * 24 * 60 * 60 * 1000)
          : undefined,
      type: status.license?.type,
    };
  }

  /**
   * Check if the app is licensed (convenience method)
   */
  async isLicensed(): Promise<boolean> {
    const status = await this.checkLicenseStatus();
    return status.valid;
  }

  /**
   * Get Patreon URL for subscription
   */
  getPatreonUrl(): string {
    return this.config.patreonUrl;
  }

  /**
   * Get GitHub releases URL
   */
  getGitHubReleasesUrl(): string {
    return this.config.githubReleasesUrl;
  }

  /**
   * Clear all license data (for debugging/reset)
   */
  async resetLicenseData(): Promise<void> {
    localStorage.removeItem(STORAGE_KEYS.trial);
    localStorage.removeItem(STORAGE_KEYS.license);
    localStorage.removeItem(STORAGE_KEYS.machine);
    this.cachedLicense = null;
    this.cachedTrial = null;
    this.initialized = false;
  }

  /**
   * Get machine ID (for support/debugging)
   */
  async getCurrentMachineId(): Promise<string | null> {
    await this.initialize();
    return this.machineId;
  }
}

// Singleton instance
export const licenseManager = new LicenseManager();

// Export class for custom instances
export { LicenseManager };
