/**
 * Subscription Service - Trust Model
 * 
 * Simple honest trial system:
 * - 7-day free trial (local storage only)
 * - After trial: "Please support" reminder (trust-based)
 * - User can continue using without enforcement
 * - Dev bypass for private builds
 */

// Storage keys
const STORAGE_KEYS = {
  trialStart: '_mimic_trial_start_v1',
  devBypass: '_mimic_dev_bypass',
  nagDismissed: '_mimic_nag_dismissed_v1',
};

export type SubscriptionStatus = 
  | 'trial_active'
  | 'trial_expired'      // Expired but can continue (trust model)
  | 'supporter';         // User indicated they support

export interface SubscriptionState {
  status: SubscriptionStatus;
  daysRemaining: number;
  trialTotalDays: number;
  isTrialExpired: boolean;
  message: string;
  showNag: boolean;
}

/**
 * Check if running in dev/private mode
 */
export function isDevMode(): boolean {
  return localStorage.getItem(STORAGE_KEYS.devBypass) === 'true';
}

/**
 * Enable dev mode (private builds)
 */
export function enableDevMode(): void {
  localStorage.setItem(STORAGE_KEYS.devBypass, 'true');
}

/**
 * Start trial period
 */
function startTrial(): void {
  const trialData = {
    startedAt: new Date().toISOString(),
    version: '1.4.0',
  };
  localStorage.setItem(STORAGE_KEYS.trialStart, JSON.stringify(trialData));
}

/**
 * Check trial status
 */
function checkTrialStatus(): { 
  active: boolean; 
  daysRemaining: number; 
  daysElapsed: number;
  expired: boolean;
} {
  const stored = localStorage.getItem(STORAGE_KEYS.trialStart);
  
  if (!stored) {
    // First launch - start trial
    startTrial();
    return { active: true, daysRemaining: 7, daysElapsed: 0, expired: false };
  }
  
  try {
    const data = JSON.parse(stored);
    const startDate = new Date(data.startedAt);
    const now = new Date();
    const diffTime = now.getTime() - startDate.getTime();
    const daysElapsed = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    const trialTotalDays = 7;
    const daysRemaining = Math.max(0, trialTotalDays - daysElapsed);
    
    return {
      active: daysRemaining > 0,
      daysRemaining,
      daysElapsed,
      expired: daysRemaining <= 0,
    };
  } catch {
    // Corrupted data - restart trial
    startTrial();
    return { active: true, daysRemaining: 7, daysElapsed: 0, expired: false };
  }
}

/**
 * Mark user as supporter (they clicked Patreon or said they support)
 */
export function markAsSupporter(): void {
  localStorage.setItem(STORAGE_KEYS.nagDismissed, 'true');
}

/**
 * Check if user has dismissed nag
 */
export function hasDismissedNag(): boolean {
  return localStorage.getItem(STORAGE_KEYS.nagDismissed) === 'true';
}

/**
 * Main subscription check - Trust Model
 */
export async function checkSubscription(): Promise<SubscriptionState> {
  const trialTotalDays = 7;
  
  // Dev mode = always active, no nag
  if (isDevMode()) {
    return {
      status: 'supporter',
      daysRemaining: 999,
      trialTotalDays,
      isTrialExpired: false,
      message: 'Developer mode active.',
      showNag: false,
    };
  }
  
  const trialInfo = checkTrialStatus();
  const nagDismissed = hasDismissedNag();
  
  // Trial still active
  if (trialInfo.active) {
    return {
      status: 'trial_active',
      daysRemaining: trialInfo.daysRemaining,
      trialTotalDays,
      isTrialExpired: false,
      message: `Trial active - ${trialInfo.daysRemaining} day${trialInfo.daysRemaining !== 1 ? 's' : ''} remaining. Consider supporting on Patreon!`,
      showNag: false,
    };
  }
  
  // Trial expired - trust model: allow continue, show optional nag
  return {
    status: 'trial_expired',
    daysRemaining: 0,
    trialTotalDays,
    isTrialExpired: true,
    message: 'Your 7-day trial has ended. Please consider supporting on Patreon to help development.',
    showNag: !nagDismissed,
  };
}

/**
 * Reset trial (for testing)
 */
export function resetTrial(): void {
  localStorage.removeItem(STORAGE_KEYS.trialStart);
  localStorage.removeItem(STORAGE_KEYS.nagDismissed);
}

/**
 * Get Patreon URL
 */
export function getPatreonUrl(): string {
  return 'https://www.patreon.com/c/MimicAIDigitalAssistant';
}

/**
 * Get GitHub releases URL
 */
export function getGitHubReleasesUrl(): string {
  return 'https://github.com/yourusername/mimic-ai/releases';
}
