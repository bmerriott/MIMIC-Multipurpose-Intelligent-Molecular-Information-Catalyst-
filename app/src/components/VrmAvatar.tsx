import { useRef, useEffect, useState, memo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { VRM, VRMLoaderPlugin } from "@pixiv/three-vrm";
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from "@pixiv/three-vrm-animation";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { analyzeQwenAudio, LipSyncPlayer } from "@/services/audioAnalysis";
import { audioAnalyzer } from "@/services/audioAnalyzer";
import { useStore } from "@/store";
import type { Persona } from "@/types";

// ============================================
// TYPES
// ============================================
export interface VrmAvatarProps {
  modelUrl: string;
  isListening: boolean;
  isSpeaking: boolean;
  primaryColor: string;
  enabledBaseVrmas?: string[];
  personaVrmas?: Record<string, string>;
  autoAnimationEnabled?: boolean;
  persona?: Persona; // NEW: For personality integration
  onPositionUpdate?: (position: THREE.Vector3) => void; // NEW: Report position for snap-back
  userControllingCamera?: boolean; // NEW: When true, avatar won't auto-face camera
}

// ============================================
// CONSTANTS
// ============================================

// Idle animations (looping)
const IDLE_ANIMATIONS = {
  breathing: "./vrma/Idle/Idle_Breathing.vrma",
  walking: "./vrma/Idle/Idle_walking.vrma",
  staring: "./vrma/Idle/Idle_staring.vrma",
  lookAround: "./vrma/Idle/Idle_Look_Around.vrma",
  rightPeekaboo: "./vrma/Idle/08_right_side_peekaboo.vrma",
};

// Emote animations (one-shot)
export const BASE_VRMA_PATHS: Record<string, string> = {
  greeting: "./vrma/Greet.vrma",
  peaceSign: "./vrma/VRMA_03.vrma",
  shoot: "./vrma/VRMA_04.vrma",
  spin: "./vrma/VRMA_05.vrma",
  modelPose: "./vrma/VRMA_06.vrma",
  squat: "./vrma/VRMA_07.vrma",
  cossackDance: "./vrma/06_CossackDance.vrma",
  peace2: "./vrma/06_peace2.vrma",
  fallingDown: "./vrma/10_falling_down_defeated.vrma",
  takaPose: "./vrma/10_takaPose.vrma",
  jikaiyokoku: "./vrma/12_jikaiyokoku.vrma",
  angelTaisou: "./vrma/13_angelTaisou_OneLoop_01.vrma",
  ctc: "./vrma/13_CTC.vrma",
  crossLegged: "./vrma/14_cross_legged.vrma",
  sanpai: "./vrma/15_sanpai.vrma",
  highFive: "./vrma/17_high_five.vrma",
  umauma: "./vrma/18_umauma.vrma",
  dance1: "./vrma/Dance_1.vrma",
};

const DEFAULT_ENABLED_EMOTES = Object.keys(BASE_VRMA_PATHS);

// Timing
const EMOTE_INTERVAL = 20;
const WALK_DURATION_MIN = 8;
const WALK_DURATION_MAX = 15;
const STARE_DURATION_MIN = 4;
const STARE_DURATION_MAX = 7;
const SCREEN_STARE_INTERVAL = 120;
const FACE_CAMERA_DURATION = 1.0;

// Movement
const WALK_SPEED = 0.4;
const WALK_BOUNDARY = 1.2;  // Reduced to keep avatar in view

// Lip sync tuning
const LIP_SYNC_CONFIG = {
  // Mouth opening range
  minMouthOpen: 0.05,             // Minimum (closed-ish)
  maxMouthOpen: 0.9,              // Maximum
  // Smoothing
  openSpeed: 0.5,                 // How fast mouth opens
  closeSpeed: 0.7,                // FAST closing for syllable separation
};

// Logger
const log = (msg: string, ...args: any[]) => {
  const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
  console.log(`[${timestamp}] [VrmAvatar] ${msg}`, ...args);
};

// ============================================
// MAIN COMPONENT
// ============================================

export const VrmAvatar = memo(function VrmAvatar({ 
  modelUrl, 
  isListening, 
  isSpeaking, 
  primaryColor,
  enabledBaseVrmas = DEFAULT_ENABLED_EMOTES,
  personaVrmas = {},
  autoAnimationEnabled = true,
  persona,
  onPositionUpdate,
  userControllingCamera = false,
}: VrmAvatarProps) {
  const groupRef = useRef<THREE.Group>(null);
  const vrmRef = useRef<VRM | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { camera } = useThree();
  
  // Get TTS text from store for Browser TTS lip sync
  const ttsText = useStore(state => state.ttsText);
  const ttsAudioData = useStore(state => state.ttsAudioData);
  
  // Animation refs
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const clipsRef = useRef<Map<string, THREE.AnimationClip>>(new Map());
  const currentActionRef = useRef<THREE.AnimationAction | null>(null);
  const currentAnimNameRef = useRef<string | null>(null);
  
  // State machine
  type State = 'loading' | 'launch' | 'idle' | 'walking' | 'staring' | 'emote' | 'peekaboo' | 
               'speaking' | 'screenStare' | 'firstPromptGreet' | 'faceCamera';
  const stateRef = useRef<State>('loading');
  
  // Pending animation after facing camera
  const pendingAnimRef = useRef<{name: string, loop: boolean, state: State} | null>(null);
  
  // Timers
  const nextEmoteTimeRef = useRef(0);
  const stateEndTimeRef = useRef(0);
  const nextScreenStareTimeRef = useRef(0);
  const hasDoneFirstGreetRef = useRef(false);
  
  // Track interaction time
  const lastInteractionTimeRef = useRef(Date.now() / 1000);
  
  // Track if user is controlling camera
  const userControllingCameraRef = useRef(userControllingCamera);
  
  // Update ref when prop changes
  useEffect(() => {
    userControllingCameraRef.current = userControllingCamera;
  }, [userControllingCamera]);
  
  // Position/orientation
  const currentPosRef = useRef(new THREE.Vector3(0, 0, 0));  // Start at center
  const walkTargetRef = useRef(new THREE.Vector3(0, 0, 0.5));
  
  // Model height for camera adjustment
  const modelHeightRef = useRef<number>(0);
  const cameraAdjustedRef = useRef(false);
  
  // Procedural animation
  const blinkTimerRef = useRef(0);
  const nextBlinkTimeRef = useRef(2 + Math.random() * 3);
  
  // Lip sync - Unified audio analysis system
  const mouthOpenRef = useRef(0);
  const expressionsRef = useRef<Set<string>>(new Set());
  const lipSyncPlayerRef = useRef<LipSyncPlayer | null>(null);
  const currentAmplitudeRef = useRef(0);
  const lastTtsTextRef = useRef('');
  const lastTtsAudioDataRef = useRef<string | null>(null);
  
  // Get personality traits
  const traits = persona?.avatar_personality?.traits;
  // Initialize lip sync player
  useEffect(() => {
    lipSyncPlayerRef.current = new LipSyncPlayer();
    return () => {
      lipSyncPlayerRef.current?.dispose();
    };
  }, []);
  
  // Subscribe to audio analyzer for real-time lip sync (Qwen/Kitten TTS)
  useEffect(() => {
    if (!isSpeaking || !ttsAudioData) return; // Only for audio-based TTS
    
    log('[LipSync] Subscribing to audio analyzer');
    
    // Subscribe to real-time audio amplitude from GlobalAudioPlayer
    const unsubscribe = audioAnalyzer.subscribe((amplitude) => {
      currentAmplitudeRef.current = amplitude;
    });
    
    return () => {
      unsubscribe();
      currentAmplitudeRef.current = 0;
    };
  }, [isSpeaking, ttsAudioData]);
  
  // Handle lip sync for all TTS types
  useEffect(() => {
    const setupLipSync = async () => {
      if (!isSpeaking || !ttsText) return;
      
      // Avoid re-setting up if same speech
      if (ttsText === lastTtsTextRef.current && ttsAudioData === lastTtsAudioDataRef.current) {
        return;
      }
      
      lastTtsTextRef.current = ttsText;
      lastTtsAudioDataRef.current = ttsAudioData;
      
      let frames;
      if (ttsAudioData) {
        // Qwen/Kitten TTS: Pre-analyze audio for accurate visemes
        log('[LipSync] Analyzing audio data for visemes...');
        frames = await analyzeQwenAudio(ttsAudioData);
      } else {
        // Browser TTS: Generate frames from text
        log('[LipSync] Generating text-based frames for Browser TTS...');
        const { generateBrowserTTSLipSync } = await import('@/services/audioAnalysis');
        frames = generateBrowserTTSLipSync(ttsText);
      }
      
      if (frames.length > 0 && lipSyncPlayerRef.current) {
        lipSyncPlayerRef.current.loadFrames(frames);
        
        if (ttsAudioData) {
          // For Qwen/Kitten: Audio is already playing when we get here (see ChatPanel)
          // Start lip sync immediately
          lipSyncPlayerRef.current.start((amplitude) => {
            currentAmplitudeRef.current = amplitude;
          });
          log('[LipSync] Audio-based TTS - started', frames.length, 'frames');
        } else {
          // For Browser TTS: Start immediately since we can't detect audio start
          lipSyncPlayerRef.current.start((amplitude) => {
            currentAmplitudeRef.current = amplitude;
          });
          log('[LipSync] Browser TTS - started', frames.length, 'frames');
        }
      }
    };
    
    if (isSpeaking) {
      setupLipSync();
    } else {
      lipSyncPlayerRef.current?.stop();
      lastTtsTextRef.current = '';
      lastTtsAudioDataRef.current = null;
    }
  }, [isSpeaking, ttsText, ttsAudioData]);

  log('Render state:', stateRef.current, 'speaking:', isSpeaking, 'persona:', persona?.name);

  // Load VRM
  useEffect(() => {
    if (!modelUrl || vrmRef.current) {
      log('Skipping load - already loaded or no URL');
      return;
    }
    
    log('=== LOADING VRM ===');
    setIsLoaded(false);
    setError(null);
    cameraAdjustedRef.current = false;

    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    loader.load(
      modelUrl,
      async (gltf) => {
        const vrm = gltf.userData.vrm as VRM;
        if (!vrm) {
          setError("No VRM data");
          return;
        }

        log('VRM loaded');
        vrm.scene.traverse((obj) => { obj.frustumCulled = false; });

        // Calculate model height for camera adjustment
        if (vrm.humanoid) {
          const box = new THREE.Box3().setFromObject(vrm.scene);
          const height = box.max.y - box.min.y;
          modelHeightRef.current = height;
          
          // Position feet at y=0
          vrm.scene.position.y = -box.min.y;
          
          log('Model height:', height.toFixed(2), 'meters');
        }

        if (vrm.expressionManager) {
          expressionsRef.current = new Set(vrm.expressionManager.expressions.map(e => e.expressionName));
          log('Expressions:', Array.from(expressionsRef.current));
        }

        vrmRef.current = vrm;
        
        if (groupRef.current) {
          groupRef.current.clear();
          groupRef.current.add(vrm.scene);
          groupRef.current.position.copy(currentPosRef.current);
          groupRef.current.rotation.y = Math.PI;
        }

        const mixer = new THREE.AnimationMixer(vrm.scene);
        mixerRef.current = mixer;

        // Load animations
        const allPaths = { ...IDLE_ANIMATIONS, ...getEmotePaths(enabledBaseVrmas, personaVrmas) };
        const loadedClips = new Map<string, THREE.AnimationClip>();
        
        for (const [name, path] of Object.entries(allPaths)) {
          try {
            const animLoader = new GLTFLoader();
            animLoader.register((parser) => new VRMAnimationLoaderPlugin(parser));
            const animGltf = await animLoader.loadAsync(path);
            if (animGltf.userData.vrmAnimations?.[0]) {
              const clip = createVRMAnimationClip(animGltf.userData.vrmAnimations[0], vrm);
              loadedClips.set(name, clip);
            }
          } catch (err) {
            log(`Failed: ${name}`);
          }
        }
        
        clipsRef.current = loadedClips;
        log('Loaded', loadedClips.size, 'animations');
        
        // Start with launch - face camera first, then greet
        const now = Date.now() / 1000;
        stateRef.current = 'launch';
        pendingAnimRef.current = { name: 'greeting', loop: false, state: 'walking' };
        stateEndTimeRef.current = now + FACE_CAMERA_DURATION;
        nextScreenStareTimeRef.current = now + SCREEN_STARE_INTERVAL;
        nextEmoteTimeRef.current = now + EMOTE_INTERVAL;
        
        setIsLoaded(true);
      },
      undefined,
      (err) => {
        log('Failed to load VRM:', err);
        setError("Failed to load VRM");
      }
    );
  }, [modelUrl, enabledBaseVrmas, personaVrmas]);

  // Adjust camera based on model height (instead of moving model)
  useEffect(() => {
    if (!isLoaded || cameraAdjustedRef.current || modelHeightRef.current === 0) return;
    
    const height = modelHeightRef.current;
    
    // Adjust camera distance based on model height
    // Keep camera further back to ensure full avatar is visible
    const baseDistance = 4.5; // Increased from 2.5 to keep avatar fully in frame
    const heightFactor = Math.max(0, (height - 1.5) * 0.8); // Adjust for models taller than 1.5m
    const targetDistance = baseDistance + heightFactor;
    
    // Only adjust if camera is too close (don't override user's zoom out)
    const currentDistance = camera.position.distanceTo(new THREE.Vector3(0, 0, 0));
    if (currentDistance < targetDistance - 0.5) {
      log('Adjusting camera for model height:', height.toFixed(2), '-> distance:', targetDistance.toFixed(2));
      
      // Move camera back while maintaining angle
      const direction = camera.position.clone().normalize();
      const newPosition = direction.multiplyScalar(targetDistance);
      newPosition.y = Math.max(1.5, height * 0.5); // Position at mid-body
      camera.position.copy(newPosition);
    }
    
    cameraAdjustedRef.current = true;
  }, [isLoaded, camera]);

  // Play animation helper
  const playAnim = (name: string, loop: boolean) => {
    if (!mixerRef.current) return false;
    if (currentAnimNameRef.current === name) return true;
    
    const clip = clipsRef.current.get(name);
    if (!clip) {
      log('Animation not found:', name);
      return false;
    }
    
    if (currentActionRef.current) currentActionRef.current.fadeOut(0.3);
    
    const action = mixerRef.current.clipAction(clip);
    action.reset().fadeIn(0.3);
    action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce, loop ? Infinity : 1);
    if (!loop) action.clampWhenFinished = true;
    action.play();
    
    currentActionRef.current = action;
    currentAnimNameRef.current = name;
    log('Play:', name, loop ? '(loop)' : '(once)');
    return true;
  };

  const stopAnim = () => {
    if (currentActionRef.current) {
      currentActionRef.current.fadeOut(0.3);
      currentActionRef.current = null;
      currentAnimNameRef.current = null;
    }
  };

  // Start an animation - first faces camera, then plays
  const startAnimation = (animName: string, loop: boolean, nextState: State, instant: boolean = false) => {
    if (instant) {
      playAnim(animName, loop);
      stateRef.current = nextState;
      return;
    }
    
    pendingAnimRef.current = { name: animName, loop, state: nextState };
    stateRef.current = 'faceCamera';
    stateEndTimeRef.current = Date.now() / 1000 + FACE_CAMERA_DURATION;
    log('Facing camera before:', animName);
  };

  // Expose playEmote to window
  useEffect(() => {
    if (!isLoaded) return;
    
    (window as any).vrmPlayEmote = (name: string) => {
      if (!clipsRef.current.has(name)) return false;
      log('Manual emote:', name);
      startAnimation(name, false, 'emote');
      return true;
    };
    
    (window as any).vrmGetEmotes = () => Array.from(clipsRef.current.keys()).filter(k => 
      !['breathing', 'walking', 'staring', 'lookAround', 'rightPeekaboo'].includes(k)
    );
    
    return () => {
      delete (window as any).vrmPlayEmote;
      delete (window as any).vrmGetEmotes;
    };
  }, [isLoaded]);



  // Get rotation to face camera
  // VRM models typically face +Z, so we need to rotate to face the camera direction
  const getRotationToFaceCamera = (): number => {
    const cameraPos = camera.position.clone();
    cameraPos.y = 0;
    const direction = new THREE.Vector3().subVectors(cameraPos, currentPosRef.current);
    const angleToCamera = Math.atan2(direction.x, direction.z);
    // Add PI to make model face TOWARD camera (models typically face +Z)
    return angleToCamera + Math.PI;
  };

  // Animation loop
  useFrame((state, delta) => {
    if (!vrmRef.current || !isLoaded || !groupRef.current) return;

    const vrm = vrmRef.current;
    const now = Date.now() / 1000;
    const dt = Math.min(delta, 0.1);

    vrm.update?.(state.clock.elapsedTime);

    // ============================================
    // LIP SYNC SYSTEM - Unified Audio Analysis
    // ============================================
    if (isSpeaking) {
      // Get amplitude from lip sync player (works for both Qwen and Browser TTS)
      const amplitude = currentAmplitudeRef.current;
      
      // Map amplitude to mouth opening with smoothing
      const expressiveness = traits?.expressiveness ?? 0.5;
      const range = LIP_SYNC_CONFIG.maxMouthOpen - LIP_SYNC_CONFIG.minMouthOpen;
      const targetOpen = LIP_SYNC_CONFIG.minMouthOpen + (amplitude * range * (0.5 + expressiveness));
      
      // Smooth transition
      const speed = amplitude > mouthOpenRef.current ? LIP_SYNC_CONFIG.openSpeed : LIP_SYNC_CONFIG.closeSpeed;
      mouthOpenRef.current += (targetOpen - mouthOpenRef.current) * speed;
    } else {
      // NOT SPEAKING - Close mouth quickly
      mouthOpenRef.current += (0 - mouthOpenRef.current) * LIP_SYNC_CONFIG.closeSpeed;
    }

    // Apply lip sync to VRM
    if (vrm.expressionManager) {
      const exprNames = ['aa', 'a', 'oh', 'mouthOpen', 'A', 'I', 'U', 'E', 'O'];
      const expr = exprNames.find(e => expressionsRef.current.has(e));
      if (expr) {
        vrm.expressionManager.setValue(expr, mouthOpenRef.current);
      }
    }

    // Camera position for look-at
    const cameraPos = camera.position.clone();
    cameraPos.y = 0;

    // State machine
    switch (stateRef.current) {
      case 'launch': {
        // Only auto-rotate during launch if user is not controlling camera
        let arrived = true;
        if (!userControllingCameraRef.current) {
          const targetRot = getRotationToFaceCamera();
          arrived = smoothRotate(groupRef.current, targetRot, 5, dt);
        }
        
        if (arrived && pendingAnimRef.current) {
          playAnim(pendingAnimRef.current.name, pendingAnimRef.current.loop);
          pendingAnimRef.current = null;
        }
        
        mixerRef.current?.update(dt);
        
        if (!pendingAnimRef.current && !currentActionRef.current?.isRunning()) {
          log('Launch complete, start walking');
          stateRef.current = 'walking';
          pickNewWalkTarget();
          stateEndTimeRef.current = now + WALK_DURATION_MIN + Math.random() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
          nextEmoteTimeRef.current = now + EMOTE_INTERVAL;
        }
        break;
      }

      case 'faceCamera': {
        const targetRot = getRotationToFaceCamera();
        const arrived = smoothRotate(groupRef.current, targetRot, 8, dt);
        
        if (currentAnimNameRef.current !== 'breathing') {
          playAnim('breathing', true);
        }
        mixerRef.current?.update(dt);
        
        if ((arrived || now >= stateEndTimeRef.current) && pendingAnimRef.current) {
          log('Now facing camera, playing:', pendingAnimRef.current.name);
          playAnim(pendingAnimRef.current.name, pendingAnimRef.current.loop);
          stateRef.current = pendingAnimRef.current.state;
          pendingAnimRef.current = null;
          stateEndTimeRef.current = Infinity;
        }
        break;
      }

      case 'walking': {
        if (isSpeaking && !hasDoneFirstGreetRef.current) {
          log('First prompt - greeting user');
          hasDoneFirstGreetRef.current = true;
          startAnimation('greeting', false, 'firstPromptGreet');
          break;
        }
        
        if (isSpeaking) {
          log('Start speaking');
          stateRef.current = 'speaking';
          break;
        }
        
        if (autoAnimationEnabled && now >= nextScreenStareTimeRef.current) {
          log('Start screen stare');
          startAnimation('staring', true, 'screenStare');
          stateEndTimeRef.current = now + STARE_DURATION_MIN + Math.random() * (STARE_DURATION_MAX - STARE_DURATION_MIN);
          break;
        }
        
        // Constrain avatar to stay in view
        constrainAvatarToView();
        
        const arrived = moveToward(dt, walkTargetRef.current, WALK_SPEED, groupRef.current, currentPosRef.current);
        
        if (currentAnimNameRef.current !== 'walking') {
          playAnim('walking', true);
        }
        mixerRef.current?.update(dt);
        
        // Camera follows avatar during walking
        updateCameraFollow(dt);
        
        if (autoAnimationEnabled && now >= nextEmoteTimeRef.current && !isListening) {
          const emotes = getEmoteNames(enabledBaseVrmas, personaVrmas);
          if (emotes.length > 0) {
            const emote = emotes[Math.floor(Math.random() * emotes.length)];
            log('Auto emote:', emote);
            startAnimation(emote, false, 'emote');
            nextEmoteTimeRef.current = now + EMOTE_INTERVAL;
            break;
          }
        }
        
        if (arrived || now >= stateEndTimeRef.current) {
          pickNewWalkTarget();
          stateEndTimeRef.current = now + WALK_DURATION_MIN + Math.random() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
        }
        break;
      }

      case 'firstPromptGreet': {
        // Only face camera if user is not controlling it
        if (!userControllingCameraRef.current) {
          const targetRot = getRotationToFaceCamera();
          smoothRotate(groupRef.current, targetRot, 5, dt);
        }
        
        mixerRef.current?.update(dt);
        
        // Camera follows during greeting
        updateCameraFollow(dt);
        
        if (!currentActionRef.current?.isRunning()) {
          log('First prompt greet complete, start speaking state');
          stateRef.current = 'speaking';
        }
        break;
      }

      case 'staring':
      case 'screenStare': {
        if (isSpeaking) {
          log('Staring interrupted - start speaking');
          stateRef.current = 'speaking';
          break;
        }
        
        // Only face camera if user is not controlling it
        if (!userControllingCameraRef.current) {
          const targetRot = getRotationToFaceCamera();
          smoothRotate(groupRef.current, targetRot, 3, dt);
        }
        
        if (currentAnimNameRef.current !== 'staring') {
          playAnim('staring', true);
        }
        mixerRef.current?.update(dt);
        
        // Camera follows during staring
        updateCameraFollow(dt);
        
        if (now >= stateEndTimeRef.current) {
          log('Done staring, walk');
          stateRef.current = 'walking';
          stopAnim();
          pickNewWalkTarget();
          stateEndTimeRef.current = now + WALK_DURATION_MIN + Math.random() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
          nextScreenStareTimeRef.current = now + SCREEN_STARE_INTERVAL;
          lastInteractionTimeRef.current = now;
        }
        break;
      }

      case 'speaking': {
        // DEBUG: Log state of isSpeaking every frame when in speaking state
        if (Math.random() < 0.02) {  // Log ~2% of frames to avoid spam
          log(`[SpeakingState] isSpeaking=${isSpeaking}, anim=${currentAnimNameRef.current}, lipSyncAmp=${currentAmplitudeRef.current.toFixed(3)}`);
        }
        
        // Check for transition OUT of speaking first
        if (!isSpeaking) {
          log('>>> Speaking ended, transitioning to walking');
          // Clear any pending animation
          pendingAnimRef.current = null;
          // Stop current animation
          stopAnim();
          // Transition to walking state FIRST
          stateRef.current = 'walking';
          // Set up walking parameters
          pickNewWalkTarget();
          stateEndTimeRef.current = now + WALK_DURATION_MIN + Math.random() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
          nextEmoteTimeRef.current = now + EMOTE_INTERVAL;
          nextScreenStareTimeRef.current = now + SCREEN_STARE_INTERVAL;
          lastInteractionTimeRef.current = now;
          // Start walking animation immediately
          log('>>> Starting walking animation after speaking');
          playAnim('walking', true);
          // Skip rest of speaking logic
          break;
        }
        
        // During speaking: Stay in place, play breathing animation, face camera
        // DO NOT move avatar or camera - this prevents zoom drift
        
        // Only face camera if user is not controlling it
        if (!userControllingCameraRef.current) {
          const targetRot = getRotationToFaceCamera();
          smoothRotate(groupRef.current, targetRot, 5, dt);
        }
        
        // Play breathing animation while speaking
        if (currentAnimNameRef.current !== 'breathing') {
          playAnim('breathing', true);
        }
        
        mixerRef.current?.update(dt);
        
        // DO NOT update camera follow during speaking - prevents zoom drift
        // Just report position for UI/debugging
        if (onPositionUpdate) {
          onPositionUpdate(groupRef.current.position);
        }
        break;
      }

      case 'emote': {
        if (isSpeaking) {
          log('Emote interrupted - start speaking');
          stateRef.current = 'speaking';
          break;
        }
        
        // Only face camera if user is not controlling it
        if (!userControllingCameraRef.current) {
          const targetRot = getRotationToFaceCamera();
          smoothRotate(groupRef.current, targetRot, 3, dt);
        }
        
        mixerRef.current?.update(dt);
        
        // Camera follows during emote
        updateCameraFollow(dt);
        
        if (!currentActionRef.current?.isRunning()) {
          log('Emote done, back to walking');
          stateRef.current = 'walking';
          stopAnim();
          // Force walking animation
          if (currentAnimNameRef.current !== 'walking') {
            playAnim('walking', true);
          }
          pickNewWalkTarget();
          stateEndTimeRef.current = now + WALK_DURATION_MIN + Math.random() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
          nextEmoteTimeRef.current = now + EMOTE_INTERVAL;  // Reset emote timer
          nextScreenStareTimeRef.current = now + SCREEN_STARE_INTERVAL;
          lastInteractionTimeRef.current = now;
        }
        break;
      }

      case 'peekaboo': {
        if (isSpeaking) {
          log('Peekaboo interrupted - start speaking');
          stateRef.current = 'speaking';
          break;
        }
        
        const peekPos = new THREE.Vector3(3, 0, 1);
        const arrived = moveToward(dt, peekPos, WALK_SPEED, groupRef.current, currentPosRef.current);
        
        if (!arrived) {
          if (currentAnimNameRef.current !== 'walking') {
            playAnim('walking', true);
          }
          mixerRef.current?.update(dt);
        } else {
          if (currentAnimNameRef.current !== 'rightPeekaboo') {
            playAnim('rightPeekaboo', false);
          }
          mixerRef.current?.update(dt);
          
          const targetRot = getRotationToFaceCamera();
          groupRef.current.rotation.y = targetRot;
          
          if (!currentActionRef.current?.isRunning()) {
            log('Peekaboo done');
            stateRef.current = 'walking';
            stopAnim();
            pickNewWalkTarget();
            nextScreenStareTimeRef.current = now + SCREEN_STARE_INTERVAL;
            lastInteractionTimeRef.current = now;
          }
        }
        break;
      }
    }

    // Procedural: blinking
    blinkTimerRef.current += dt;
    if (blinkTimerRef.current >= nextBlinkTimeRef.current) {
      blinkTimerRef.current = 0;
      nextBlinkTimeRef.current = 2 + Math.random() * 4;
      if (vrm.expressionManager) {
        vrm.expressionManager.setValue('blink', 1);
        setTimeout(() => vrm.expressionManager?.setValue('blink', 0), 150);
      }
    }
    
    // Report position for snap-back functionality
    if (onPositionUpdate) {
      onPositionUpdate(currentPosRef.current);
    }
  });

  function smoothRotate(group: THREE.Group, targetRot: number, speed: number, dt: number): boolean {
    let rotDiff = targetRot - group.rotation.y;
    while (rotDiff > Math.PI) rotDiff -= Math.PI * 2;
    while (rotDiff < -Math.PI) rotDiff += Math.PI * 2;
    
    if (Math.abs(rotDiff) < 0.05) {
      group.rotation.y = targetRot;
      return true;
    }
    
    group.rotation.y += Math.sign(rotDiff) * Math.min(Math.abs(rotDiff), speed * dt);
    return false;
  }

  function pickNewWalkTarget() {
    // Pick a position within view of the camera
    const angle = Math.random() * Math.PI * 2;
    // Keep closer to center so avatar stays visible
    const radius = 0.3 + Math.random() * (WALK_BOUNDARY * 0.6);
    walkTargetRef.current.set(
      Math.cos(angle) * radius,
      0,
      Math.sin(angle) * radius
    );
  }

  // Camera follow is now handled in AvatarScene component
  // This function is kept for compatibility but does nothing
  function updateCameraFollow(_dt: number) {
    // Camera follow logic moved to AvatarCanvas to avoid conflicts
  }

  // Reset avatar to center if it wanders too far
  function constrainAvatarToView() {
    const distanceFromCenter = Math.sqrt(
      currentPosRef.current.x ** 2 + currentPosRef.current.z ** 2
    );
    
    if (distanceFromCenter > WALK_BOUNDARY) {
      // Gently nudge back toward center
      const angle = Math.atan2(currentPosRef.current.x, currentPosRef.current.z);
      walkTargetRef.current.set(
        Math.sin(angle) * (WALK_BOUNDARY * 0.5),
        0,
        Math.cos(angle) * (WALK_BOUNDARY * 0.5)
      );
    }
  }

  function moveToward(dt: number, target: THREE.Vector3, speed: number, group: THREE.Group, currentPos: THREE.Vector3) {
    const direction = new THREE.Vector3().subVectors(target, currentPos);
    const distance = direction.length();
    
    if (distance < 0.1) return true;
    
    direction.normalize();
    const targetRot = Math.atan2(direction.x, direction.z);
    
    let rotDiff = targetRot - group.rotation.y;
    while (rotDiff > Math.PI) rotDiff -= Math.PI * 2;
    while (rotDiff < -Math.PI) rotDiff += Math.PI * 2;
    
    if (Math.abs(rotDiff) > 0.05) {
      group.rotation.y += Math.sign(rotDiff) * Math.min(Math.abs(rotDiff), 4 * dt);
    }
    
    currentPos.x += Math.sin(group.rotation.y) * speed * dt;
    currentPos.z += Math.cos(group.rotation.y) * speed * dt;
    group.position.copy(currentPos);
    
    return false;
  }

  function getEmotePaths(enabled: string[], persona: Record<string, string>) {
    const paths: Record<string, string> = {};
    for (const name of enabled) {
      if (BASE_VRMA_PATHS[name]) paths[name] = BASE_VRMA_PATHS[name];
    }
    Object.entries(persona).forEach(([name, path]) => { paths[name] = path; });
    return paths;
  }

  function getEmoteNames(enabled: string[], persona: Record<string, string>) {
    return Object.keys(getEmotePaths(enabled, persona)).filter(k => 
      !['breathing', 'walking', 'staring', 'lookAround', 'rightPeekaboo'].includes(k)
    );
  }

  if (error) {
    return (
      <group ref={groupRef}>
        <mesh><boxGeometry args={[1, 1, 1]} /><meshStandardMaterial color="red" /></mesh>
      </group>
    );
  }

  if (!isLoaded) {
    return (
      <group ref={groupRef}>
        <mesh><boxGeometry args={[0.3, 0.3, 0.3]} /><meshStandardMaterial color={primaryColor} wireframe /></mesh>
      </group>
    );
  }

  return <group ref={groupRef} scale={[1.2, 1.2, 1.2]} />;
});

export default VrmAvatar;
