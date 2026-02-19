/**
 * Voice Tuning Service
 * Manages per-persona voice tuning parameters and applies them during playback
 * 
 * Categories:
 * - Synthesis params: Require regeneration (warmth, expressiveness, stability, clarity, breathiness, resonance, emotion, emphasis, pauses, energy)
 * - Post-processing params: Applied during playback (pitch, speed, reverb, EQ, compression)
 */

import type { VoiceTuningParams } from "@/store";

// Params that require voice regeneration
export const SYNTHESIS_PARAMS: (keyof VoiceTuningParams)[] = [
  'warmth',
  'expressiveness', 
  'stability',
  'clarity',
  'breathiness',
  'resonance',
  'emotion',
  'emphasis',
  'pauses',
  'energy',
];

// Params that can be applied during playback (post-processing)
export const POST_PROCESSING_PARAMS: (keyof VoiceTuningParams)[] = [
  'pitchShift',
  'speed',
  'reverb',
  'eqLow',
  'eqMid',
  'eqHigh',
  'compression',
];

/**
 * Check if a parameter requires regeneration
 */
export function requiresRegeneration(param: keyof VoiceTuningParams): boolean {
  return SYNTHESIS_PARAMS.includes(param);
}

/**
 * Check if any synthesis params have changed
 */
export function haveSynthesisParamsChanged(
  oldParams: VoiceTuningParams,
  newParams: Partial<VoiceTuningParams>
): boolean {
  return SYNTHESIS_PARAMS.some(param => param in newParams && oldParams[param] !== newParams[param]);
}

/**
 * Get only post-processing params
 */
export function getPostProcessingParams(params: VoiceTuningParams): Pick<VoiceTuningParams, 'pitchShift' | 'speed' | 'reverb' | 'eqLow' | 'eqMid' | 'eqHigh' | 'compression'> {
  return {
    pitchShift: params.pitchShift,
    speed: params.speed,
    reverb: params.reverb,
    eqLow: params.eqLow,
    eqMid: params.eqMid,
    eqHigh: params.eqHigh,
    compression: params.compression,
  };
}

/**
 * Get only synthesis params
 */
export function getSynthesisParams(params: VoiceTuningParams): Pick<VoiceTuningParams, 'warmth' | 'expressiveness' | 'stability' | 'clarity' | 'breathiness' | 'resonance' | 'emotion' | 'emphasis' | 'pauses' | 'energy'> {
  return {
    warmth: params.warmth,
    expressiveness: params.expressiveness,
    stability: params.stability,
    clarity: params.clarity,
    breathiness: params.breathiness,
    resonance: params.resonance,
    emotion: params.emotion,
    emphasis: params.emphasis,
    pauses: params.pauses,
    energy: params.energy,
  };
}

/**
 * Convert store tuning params to TTS API params
 */
export function toTTSParams(params: VoiceTuningParams): Record<string, number | string> {
  return {
    // Basic
    pitch_shift: params.pitchShift,
    speed: params.speed,
    
    // Voice characteristics
    warmth: params.warmth,
    expressiveness: params.expressiveness,
    stability: params.stability,
    clarity: params.clarity,
    breathiness: params.breathiness,
    resonance: params.resonance,
    
    // Speech
    emotion: params.emotion,
    emphasis: params.emphasis,
    pauses: params.pauses,
    energy: params.energy,
    
    // Audio effects
    reverb: params.reverb,
    eq_low: params.eqLow,
    eq_mid: params.eqMid,
    eq_high: params.eqHigh,
    compression: params.compression,
  };
}

/**
 * Human-readable parameter descriptions
 */
export const PARAM_DESCRIPTIONS: Record<keyof VoiceTuningParams, { label: string; description: string; category: 'synthesis' | 'post-processing' }> = {
  warmth: { label: 'Warmth', description: 'Natural, mellow tone quality', category: 'synthesis' },
  expressiveness: { label: 'Expressiveness', description: 'Emotional variation in speech', category: 'synthesis' },
  stability: { label: 'Stability', description: 'Consistency vs creativity in voice', category: 'synthesis' },
  clarity: { label: 'Clarity', description: 'Articulation sharpness', category: 'synthesis' },
  breathiness: { label: 'Breathiness', description: 'Airiness in voice texture', category: 'synthesis' },
  resonance: { label: 'Resonance', description: 'Depth and fullness of voice', category: 'synthesis' },
  emotion: { label: 'Emotion', description: 'Base emotional tone', category: 'synthesis' },
  emphasis: { label: 'Emphasis', description: 'Word stress intensity', category: 'synthesis' },
  pauses: { label: 'Pauses', description: 'Pause length between phrases', category: 'synthesis' },
  energy: { label: 'Energy', description: 'Overall vocal energy level', category: 'synthesis' },
  pitchShift: { label: 'Pitch Shift', description: 'Raise or lower voice pitch', category: 'post-processing' },
  speed: { label: 'Speed', description: 'Speech rate multiplier', category: 'post-processing' },
  reverb: { label: 'Reverb', description: 'Room ambiance effect', category: 'post-processing' },
  eqLow: { label: 'EQ Low', description: 'Bass frequencies (100Hz)', category: 'post-processing' },
  eqMid: { label: 'EQ Mid', description: 'Mid frequencies (1kHz)', category: 'post-processing' },
  eqHigh: { label: 'EQ High', description: 'Treble frequencies (10kHz)', category: 'post-processing' },
  compression: { label: 'Compression', description: 'Dynamic range compression', category: 'post-processing' },
};
