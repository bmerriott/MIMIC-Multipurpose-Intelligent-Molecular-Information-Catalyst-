import { useRef, useEffect, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { audioAnalyzer } from "@/services/audioAnalyzer";

interface GenericAvatarProps {
  modelUrl: string;
  isListening: boolean;
  isSpeaking: boolean;
  primaryColor: string;
  onApproachComplete?: () => void;
}

// Room for wandering
const ROOM_SIZE = 8;
const WALK_SPEED = 0.25; // Slower walking

// Look-around directions
const LOOK_DIRECTIONS = [
  { x: 0, y: 0, label: 'center' },
  { x: 0.2, y: -0.4, label: 'left' },
  { x: 0.1, y: 0.4, label: 'right' },
  { x: -0.3, y: 0, label: 'up' },
  { x: 0.3, y: 0, label: 'down' },
  { x: 0.2, y: -0.6, label: 'far_left' },
  { x: 0.1, y: 0.6, label: 'far_right' },
];

type AvatarState = 
  | 'idle' 
  | 'wandering' 
  | 'approaching' 
  | 'speaking' 
  | 'listening' 
  | 'thinking' 
  | 'inspecting'
  | 'gesturing'
  | 'peekaboo'
  | 'sneaking'
  | 'looking_around'
  | 'knocking';

// Bone name mappings - handles common naming conventions
const BONE_MAPPINGS: Record<string, string[]> = {
  hips: ['hips', 'Hips', 'hip', 'Hip', 'pelvis', 'Pelvis'],
  spine: ['spine', 'Spine', 'torso', 'Torso'],
  chest: ['chest', 'Chest', 'upperChest', 'UpperChest', 'spine2', 'Spine2'],
  neck: ['neck', 'Neck'],
  head: ['head', 'Head'],
  leftUpperArm: ['leftUpperArm', 'LeftUpperArm', 'left_shoulder', 'LeftShoulder', 'upperarm_l', 'UpperArm_L'],
  rightUpperArm: ['rightUpperArm', 'RightUpperArm', 'right_shoulder', 'RightShoulder', 'upperarm_r', 'UpperArm_R'],
  leftLowerArm: ['leftLowerArm', 'LeftLowerArm', 'left_elbow', 'LeftElbow', 'lowerarm_l', 'LowerArm_L', 'forearm_l'],
  rightLowerArm: ['rightLowerArm', 'RightLowerArm', 'right_elbow', 'RightElbow', 'lowerarm_r', 'LowerArm_R', 'forearm_r'],
  leftHand: ['leftHand', 'LeftHand', 'hand_l', 'Hand_L'],
  rightHand: ['rightHand', 'RightHand', 'hand_r', 'Hand_R'],
  leftUpperLeg: ['leftUpperLeg', 'LeftUpperLeg', 'left_hip', 'LeftHip', 'thigh_l', 'Thigh_L'],
  rightUpperLeg: ['rightUpperLeg', 'RightUpperLeg', 'right_hip', 'RightHip', 'thigh_r', 'Thigh_R'],
  leftLowerLeg: ['leftLowerLeg', 'LeftLowerLeg', 'left_knee', 'LeftKnee', 'shin_l', 'Shin_L', 'calf_l'],
  rightLowerLeg: ['rightLowerLeg', 'RightLowerLeg', 'right_knee', 'RightKnee', 'shin_r', 'Shin_R', 'calf_r'],
  leftFoot: ['leftFoot', 'LeftFoot', 'foot_l', 'Foot_L'],
  rightFoot: ['rightFoot', 'RightFoot', 'foot_r', 'Foot_R'],
};

interface Pose {
  leftArmZ: number; leftArmX: number;
  rightArmZ: number; rightArmX: number;
  leftForearmX: number; rightForearmX: number;
  leftHandZ: number; rightHandZ: number;
  headX: number; headY: number; headZ: number;
  neckY: number;
  chestY: number;
  spineX: number; spineZ: number;
}

// Generate random idle/wander times
const getRandomIdleTime = () => 3 + Math.random() * 5;
const getRandomWanderTime = () => 8 + Math.random() * 10;

export function GenericAvatar({ 
  modelUrl, 
  isListening, 
  isSpeaking, 
  primaryColor,
  onApproachComplete
}: GenericAvatarProps) {
  const groupRef = useRef<THREE.Group>(null);
  const gltfRef = useRef<GLTF | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const audioAmplitudeRef = useRef(0);
  const smoothedAmplitudeRef = useRef(0);
  const silenceTimerRef = useRef(0);
  const isSilentRef = useRef(false);
  const hasApproachedRef = useRef(false);
  const lastInteractionTimeRef = useRef(Date.now());

  // Discovered bones
  const bonesRef = useRef<Map<string, THREE.Object3D>>(new Map());
  // Discovered morph targets
  const morphTargetsRef = useRef<Map<string, { mesh: THREE.Mesh; index: number }>>(new Map());
  
  const navRef = useRef({
    position: new THREE.Vector3(0, -1, 0),
    targetPosition: new THREE.Vector3(0, -1, 0),
    rotation: 0,
    targetRotation: 0,
    isMoving: false,
    stateTimer: 0,
    currentState: 'idle' as AvatarState,
    previousState: 'idle' as AvatarState,
    walkCycle: 0,
    gesturePhase: 0,
    sneakProgress: 0,
    peekProgress: 0,
    // Look around state
    lookTarget: { x: 0, y: 0 },
    lookTimer: 0,
    lookDuration: 1.5,
    isLookingAtScreen: false,
    // Knock state
    knockPhase: 0,
    knockCount: 0,
    // Idle breathing
    breathOffset: Math.random() * 100,
  });

  const currentPoseRef = useRef<Pose>({
    leftArmZ: 0.1, leftArmX: 0,
    rightArmZ: -0.1, rightArmX: 0,
    leftForearmX: -0.2, rightForearmX: -0.2,
    leftHandZ: 0, rightHandZ: 0,
    headX: 0, headY: 0, headZ: 0,
    neckY: 0,
    chestY: 0,
    spineX: 0, spineZ: 0,
  });

  useEffect(() => {
    const unsubscribe = audioAnalyzer.subscribe((amp) => {
      audioAmplitudeRef.current = amp;
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (isSpeaking && !hasApproachedRef.current) {
      navRef.current.currentState = 'approaching';
      hasApproachedRef.current = true;
    }
    if (!isSpeaking) {
      hasApproachedRef.current = false;
    }
    // Update last interaction time
    if (isSpeaking || isListening) {
      lastInteractionTimeRef.current = Date.now();
    }
  }, [isSpeaking, isListening]);

  // Auto-discover bones and morph targets from loaded model
  const discoverModelFeatures = (gltf: GLTF) => {
    const bones = new Map<string, THREE.Object3D>();
    const morphTargets = new Map<string, { mesh: THREE.Mesh; index: number }>();

    gltf.scene.traverse((obj) => {
      // Find bones by name
      if (obj.type === 'Bone' || obj.name.toLowerCase().includes('bone')) {
        for (const [standardName, possibleNames] of Object.entries(BONE_MAPPINGS)) {
          if (!bones.has(standardName) && possibleNames.includes(obj.name)) {
            bones.set(standardName, obj);
            console.log(`[GenericAvatar] Found bone: ${standardName} -> ${obj.name}`);
          }
        }
      }

      // Also check for skinned mesh skeleton bones
      if (obj instanceof THREE.SkinnedMesh && obj.skeleton) {
        obj.skeleton.bones.forEach((bone) => {
          for (const [standardName, possibleNames] of Object.entries(BONE_MAPPINGS)) {
            if (!bones.has(standardName) && possibleNames.includes(bone.name)) {
              bones.set(standardName, bone);
              console.log(`[GenericAvatar] Found skeleton bone: ${standardName} -> ${bone.name}`);
            }
          }
        });

        // Find morph targets (shape keys)
        if (obj.morphTargetDictionary) {
          Object.entries(obj.morphTargetDictionary).forEach(([name, index]) => {
            const lowerName = name.toLowerCase();
            // Map common morph target names
            const mappings: Record<string, string[]> = {
              'mouthOpen': ['mouthopen', 'mouth_open', 'jawopen', 'jaw_open', 'a', 'aa'],
              'happy': ['happy', 'smile', 'joy'],
              'blink': ['blink', 'eyeblink', 'eyesclosed', 'eyes_closed'],
              'sad': ['sad', 'sorrow'],
              'angry': ['angry', 'anger'],
              'surprised': ['surprised', 'surprise', 'shock'],
              'ih': ['ih', 'i', 'ee'],
              'ou': ['ou', 'u', 'oo'],
            };

            for (const [standard, variants] of Object.entries(mappings)) {
              if (!morphTargets.has(standard) && variants.includes(lowerName)) {
                morphTargets.set(standard, { mesh: obj, index });
                console.log(`[GenericAvatar] Found morph target: ${standard} -> ${name}`);
              }
            }
            
            // Also store by original name
            if (!morphTargets.has(name)) {
              morphTargets.set(name, { mesh: obj, index });
            }
          });
        }
      }
    });

    bonesRef.current = bones;
    morphTargetsRef.current = morphTargets;
  };

  const setMorphValue = (name: string, value: number) => {
    const target = morphTargetsRef.current.get(name);
    if (target) {
      target.mesh.morphTargetInfluences![target.index] = value;
    }
  };

  const resetAllMorphs = () => {
    morphTargetsRef.current.forEach(({ mesh, index }) => {
      if (mesh.morphTargetInfluences) {
        mesh.morphTargetInfluences[index] = 0;
      }
    });
  };

  useEffect(() => {
    if (!modelUrl) return;
    setIsLoaded(false);
    setError(null);
    bonesRef.current.clear();
    morphTargetsRef.current.clear();

    const loader = new GLTFLoader();

    loader.load(
      modelUrl,
      (gltf) => {
        gltf.scene.traverse((obj) => {
          obj.frustumCulled = false;
        });

        discoverModelFeatures(gltf);
        gltfRef.current = gltf;

        if (groupRef.current) {
          while (groupRef.current.children.length > 0) {
            groupRef.current.remove(groupRef.current.children[0]);
          }
          groupRef.current.add(gltf.scene);
        }

        setIsLoaded(true);
      },
      undefined,
      (err) => {
        console.error("[GenericAvatar] Load error:", err);
        setError("Failed to load model");
      }
    );

    return () => {
      if (gltfRef.current) {
        gltfRef.current.scene.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry?.dispose();
            if (Array.isArray(obj.material)) {
              obj.material.forEach(m => m?.dispose?.());
            } else {
              obj.material?.dispose?.();
            }
          }
        });
        gltfRef.current = null;
      }
    };
  }, [modelUrl]);

  const pickRandomPoint = () => {
    const angle = Math.random() * Math.PI * 2;
    const radius = 2 + Math.random() * (ROOM_SIZE - 2); // Keep away from center
    return new THREE.Vector3(
      Math.cos(angle) * radius,
      -1,
      Math.sin(angle) * radius
    );
  };

  const pickRandomLookTarget = () => {
    const rand = Math.random();
    // 30% chance to look at screen
    if (rand < 0.3) {
      navRef.current.isLookingAtScreen = true;
      return { x: 0, y: 0 };
    }
    navRef.current.isLookingAtScreen = false;
    // Pick random direction
    const dir = LOOK_DIRECTIONS[Math.floor(Math.random() * LOOK_DIRECTIONS.length)];
    return { x: dir.x, y: dir.y };
  };

  const getBone = (name: string): THREE.Object3D | undefined => {
    return bonesRef.current.get(name);
  };

  useFrame((state) => {
    if (!gltfRef.current || !isLoaded || !groupRef.current) return;

    const time = state.clock.elapsedTime;
    const delta = Math.min(state.clock.getDelta(), 0.1);
    const nav = navRef.current;

    // Check for inactivity - start wandering if idle too long
    const timeSinceInteraction = (Date.now() - lastInteractionTimeRef.current) / 1000;
    const shouldWander = timeSinceInteraction > 10 && !isSpeaking && !isListening && nav.currentState === 'idle';
    
    if (shouldWander && Math.random() < 0.01) {
      nav.currentState = 'wandering';
      nav.targetPosition = pickRandomPoint();
      nav.isMoving = true;
    }

    // === STATE MACHINE ===
    const updateState = () => {
      if (nav.currentState === 'approaching') {
        const distToCamera = nav.position.distanceTo(new THREE.Vector3(0, -1, 2));
        if (distToCamera < 0.5) {
          nav.currentState = isSpeaking ? 'speaking' : 'idle';
          nav.isMoving = false;
          onApproachComplete?.();
        }
        return;
      }

      if (isSpeaking) {
        if (nav.currentState !== 'speaking' && nav.currentState !== 'gesturing') {
          nav.currentState = Math.random() > 0.4 ? 'gesturing' : 'speaking';
          nav.stateTimer = time + 3 + Math.random() * 4;
        }
        return;
      }

      if (isListening && nav.currentState !== 'listening') {
        nav.currentState = 'listening';
        nav.stateTimer = time + 2 + Math.random() * 3;
        return;
      }

      if (time > nav.stateTimer) {
        const states: AvatarState[] = ['idle', 'wandering', 'thinking', 'inspecting', 'peekaboo', 'sneaking', 'looking_around', 'knocking'];
        const weights = [0.25, 0.25, 0.12, 0.12, 0.05, 0.05, 0.1, 0.06];
        
        const rand = Math.random();
        let cum = 0;
        for (let i = 0; i < states.length; i++) {
          cum += weights[i];
          if (rand < cum) {
            nav.previousState = nav.currentState;
            nav.currentState = states[i];
            break;
          }
        }

        // Set state duration
        if (nav.currentState === 'wandering') {
          nav.stateTimer = time + getRandomWanderTime();
        } else if (nav.currentState === 'looking_around') {
          nav.stateTimer = time + 3 + Math.random() * 4;
        } else if (nav.currentState === 'knocking') {
          nav.stateTimer = time + 3;
          nav.knockPhase = 0;
          nav.knockCount = 0;
        } else {
          nav.stateTimer = time + getRandomIdleTime();
        }

        if (nav.currentState === 'wandering') {
          nav.targetPosition = pickRandomPoint();
          nav.isMoving = true;
        } else if (nav.currentState === 'approaching' || nav.currentState === 'sneaking') {
          nav.targetPosition.set(0, -1, 2);
          nav.isMoving = true;
        } else if (nav.currentState === 'peekaboo') {
          nav.peekProgress = 0;
          nav.isMoving = false;
        } else if (nav.currentState === 'looking_around') {
          nav.lookTarget = pickRandomLookTarget();
          nav.lookTimer = time;
          nav.lookDuration = 1 + Math.random() * 2;
          nav.isMoving = false;
        } else if (nav.currentState === 'knocking') {
          nav.isMoving = false;
          nav.isLookingAtScreen = true;
        } else {
          nav.isMoving = false;
        }
      }
    };

    updateState();

    // === NAVIGATION ===
    if (nav.isMoving) {
      const direction = new THREE.Vector3().subVectors(nav.targetPosition, nav.position);
      const distance = direction.length();
      
      if (distance < 0.3) {
        nav.isMoving = false;
        if (nav.currentState === 'wandering') {
          nav.currentState = 'idle';
          nav.stateTimer = time + getRandomIdleTime();
        }
      } else {
        direction.normalize();
        nav.targetRotation = Math.atan2(direction.x, direction.z);
        
        let rotDiff = nav.targetRotation - nav.rotation;
        while (rotDiff > Math.PI) rotDiff -= Math.PI * 2;
        while (rotDiff < -Math.PI) rotDiff += Math.PI * 2;
        nav.rotation += rotDiff * 2.0 * delta;
        
        const speed = nav.currentState === 'sneaking' ? WALK_SPEED * 0.4 : WALK_SPEED;
        nav.position.x += Math.sin(nav.rotation) * speed * delta;
        nav.position.z += Math.cos(nav.rotation) * speed * delta;
        
        nav.walkCycle += delta * 3; // Slower walk cycle
      }
    } else {
      nav.walkCycle = THREE.MathUtils.lerp(nav.walkCycle, 0, delta * 2);
    }

    groupRef.current.position.copy(nav.position);
    
    if (nav.currentState === 'approaching' && !nav.isMoving) {
      groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, Math.PI, delta * 3);
    } else if (nav.currentState === 'peekaboo') {
      nav.peekProgress += delta * 0.5;
      groupRef.current.rotation.y = Math.PI + Math.sin(nav.peekProgress * Math.PI) * 0.5;
      if (nav.peekProgress > 2) nav.currentState = 'idle';
    } else if (nav.currentState === 'knocking') {
      groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, Math.PI, delta * 5);
    } else {
      groupRef.current.rotation.y = nav.rotation + Math.PI;
    }

    // === POSE CALCULATION ===
    let targetPose: Partial<Pose> = {};
    const walkAmp = 0.25;

    switch (nav.currentState) {
      case 'wandering':
      case 'approaching':
      case 'sneaking': {
        const isSneak = nav.currentState === 'sneaking';
        const armSwing = isSneak ? 0.08 : 0.2;
        
        targetPose = {
          leftArmZ: 0.05, rightArmZ: -0.05,
          leftArmX: Math.sin(nav.walkCycle) * armSwing,
          rightArmX: Math.sin(nav.walkCycle + Math.PI) * armSwing,
          leftForearmX: -0.3 + Math.sin(nav.walkCycle) * 0.1,
          rightForearmX: -0.3 + Math.sin(nav.walkCycle + Math.PI) * 0.1,
          leftHandZ: Math.sin(nav.walkCycle * 2) * 0.05,
          rightHandZ: Math.sin(nav.walkCycle * 2 + Math.PI) * 0.05,
          headX: Math.sin(nav.walkCycle * 2) * 0.015,
          headY: Math.sin(time * 0.5) * 0.03,
          headZ: 0,
          neckY: Math.sin(nav.walkCycle) * 0.02,
          chestY: Math.sin(nav.walkCycle * 2) * 0.01,
          spineX: 0, spineZ: 0,
        };

        const leftLeg = getBone('leftUpperLeg');
        const rightLeg = getBone('rightUpperLeg');
        const leftKnee = getBone('leftLowerLeg');
        const rightKnee = getBone('rightLowerLeg');
        const leftFoot = getBone('leftFoot');
        const rightFoot = getBone('rightFoot');
        const hips = getBone('hips');

        const leftLegPhase = nav.walkCycle;
        const rightLegPhase = nav.walkCycle + Math.PI;

        if (leftLeg) leftLeg.rotation.x = THREE.MathUtils.lerp(leftLeg.rotation.x, Math.sin(leftLegPhase) * walkAmp, delta * 5);
        if (rightLeg) rightLeg.rotation.x = THREE.MathUtils.lerp(rightLeg.rotation.x, Math.sin(rightLegPhase) * walkAmp, delta * 5);
        
        if (leftKnee) {
          const leftKneeBend = Math.sin(leftLegPhase - 0.5);
          leftKnee.rotation.x = THREE.MathUtils.lerp(leftKnee.rotation.x, leftKneeBend > 0 ? leftKneeBend * 0.6 : 0.05, delta * 5);
        }
        if (rightKnee) {
          const rightKneeBend = Math.sin(rightLegPhase - 0.5);
          rightKnee.rotation.x = THREE.MathUtils.lerp(rightKnee.rotation.x, rightKneeBend > 0 ? rightKneeBend * 0.6 : 0.05, delta * 5);
        }
        
        if (leftFoot) leftFoot.rotation.x = THREE.MathUtils.lerp(leftFoot.rotation.x, Math.sin(leftLegPhase - 0.8) * 0.15, delta * 5);
        if (rightFoot) rightFoot.rotation.x = THREE.MathUtils.lerp(rightFoot.rotation.x, Math.sin(rightLegPhase - 0.8) * 0.15, delta * 5);

        if (isSneak && hips) {
          hips.position.y = THREE.MathUtils.lerp(hips.position.y, -1.2, delta * 3);
        }
        break;
      }

      case 'thinking': {
        targetPose = {
          leftArmZ: 0.1, rightArmZ: -0.8,
          leftArmX: 0.1, rightArmX: -0.4,
          leftForearmX: -0.3, rightForearmX: -2.0,
          leftHandZ: 0, rightHandZ: 0.3,
          headX: 0.2, headY: 0, headZ: 0.15,
          neckY: 0,
          chestY: 0,
          spineX: 0.1, spineZ: 0,
        };
        break;
      }

      case 'inspecting': {
        const inspectBob = Math.sin(time * 2) * 0.03;
        targetPose = {
          leftArmZ: 0.3, rightArmZ: -0.3,
          leftArmX: -0.5, rightArmX: -0.5,
          leftForearmX: -0.8, rightForearmX: -0.8,
          leftHandZ: 0.1, rightHandZ: -0.1,
          headX: 0.4 + inspectBob, headY: Math.sin(time * 0.7) * 0.08,
          headZ: 0,
          neckY: 0,
          chestY: 0,
          spineX: 0.35 + inspectBob, spineZ: 0,
        };
        break;
      }

      case 'speaking': {
        const talkCycle = time * 2.5;
        const gestureSize = 0.12;
        targetPose = {
          leftArmZ: 0.2, rightArmZ: -0.2,
          leftArmX: Math.sin(talkCycle) * gestureSize,
          rightArmX: Math.sin(talkCycle + 1) * gestureSize,
          leftForearmX: -0.3 + Math.sin(talkCycle * 1.5) * 0.08,
          rightForearmX: -0.3 + Math.sin(talkCycle * 1.5 + 1) * 0.08,
          leftHandZ: Math.sin(talkCycle * 2) * 0.06,
          rightHandZ: Math.sin(talkCycle * 2 + 1) * 0.06,
          headX: Math.sin(time * 1.2) * 0.02,
          headY: Math.sin(time) * 0.03,
          headZ: Math.sin(time * 0.7) * 0.015,
          neckY: Math.sin(time * 0.8) * 0.02,
          chestY: Math.sin(time * 3) * 0.005,
          spineX: 0, spineZ: 0,
        };
        break;
      }

      case 'gesturing': {
        const gestureCycle = time * 2;
        targetPose = {
          leftArmZ: 0.4 + Math.sin(gestureCycle) * 0.12,
          rightArmZ: -0.4 - Math.sin(gestureCycle + 1) * 0.12,
          leftArmX: Math.sin(gestureCycle * 1.2) * 0.2,
          rightArmX: Math.sin(gestureCycle * 1.2 + Math.PI) * 0.2,
          leftForearmX: -0.4 + Math.sin(gestureCycle * 1.5) * 0.2,
          rightForearmX: -0.4 + Math.sin(gestureCycle * 1.5 + 1) * 0.2,
          leftHandZ: Math.sin(gestureCycle * 2) * 0.12,
          rightHandZ: Math.sin(gestureCycle * 2 + 1) * 0.12,
          headX: Math.sin(time * 1.5) * 0.03,
          headY: Math.sin(time * 1.2) * 0.04,
          headZ: Math.sin(time * 0.6) * 0.015,
          neckY: Math.sin(time) * 0.03,
          chestY: Math.sin(time * 4) * 0.008,
          spineX: Math.sin(time * 1.5) * 0.015, spineZ: 0,
        };
        break;
      }

      case 'listening': {
        targetPose = {
          leftArmZ: 0.1, rightArmZ: -0.1,
          leftArmX: 0, rightArmX: 0,
          leftForearmX: -0.25, rightForearmX: -0.25,
          leftHandZ: 0, rightHandZ: 0,
          headX: -0.1, headY: Math.sin(time * 0.5) * 0.06,
          headZ: 0.05,
          neckY: Math.sin(time * 0.4) * 0.04,
          chestY: 0,
          spineX: -0.02, spineZ: 0,
        };
        break;
      }

      case 'peekaboo': {
        const peek = Math.sin(nav.peekProgress * Math.PI);
        targetPose = {
          leftArmZ: 0.2, rightArmZ: -0.2,
          leftArmX: -0.3 * peek, rightArmX: -0.3 * peek,
          leftForearmX: -0.5, rightForearmX: -0.5,
          leftHandZ: 0, rightHandZ: 0,
          headX: -0.2 * peek, headY: 0.3 * peek,
          headZ: 0.1 * peek,
          neckY: 0.2 * peek,
          chestY: 0,
          spineX: 0, spineZ: 0.1 * peek,
        };
        break;
      }

      case 'looking_around': {
        const lookProgress = Math.min((time - nav.lookTimer) / nav.lookDuration, 1);
        const lookEase = lookProgress < 0.5 
          ? 2 * lookProgress * lookProgress 
          : 1 - Math.pow(-2 * lookProgress + 2, 2) / 2;
        
        targetPose = {
          leftArmZ: 0.1, rightArmZ: -0.1,
          leftArmX: 0, rightArmX: 0,
          leftForearmX: -0.2, rightForearmX: -0.2,
          leftHandZ: 0, rightHandZ: 0,
          headX: nav.lookTarget.x * lookEase,
          headY: nav.lookTarget.y * lookEase,
          headZ: 0,
          neckY: nav.lookTarget.y * lookEase * 0.5,
          chestY: 0,
          spineX: 0, spineZ: 0,
        };
        
        if (lookProgress >= 1 && time > nav.lookTimer + nav.lookDuration + 0.5) {
          nav.currentState = 'idle';
        }
        break;
      }

      case 'knocking': {
        nav.knockPhase += delta * 8;
        const knockCycle = Math.sin(nav.knockPhase);
        const isKnocking = knockCycle > 0.7;
        const knockProgress = Math.max(0, (knockCycle - 0.7) / 0.3);
        
        if (knockCycle > 0.95 && nav.knockCount < Math.floor(nav.knockPhase / Math.PI)) {
          nav.knockCount = Math.floor(nav.knockPhase / Math.PI);
          window.dispatchEvent(new CustomEvent('avatar-knock'));
        }
        
        targetPose = {
          leftArmZ: 0.3,
          leftArmX: -0.5 - (isKnocking ? knockProgress * 0.2 : 0),
          rightArmZ: -0.1,
          rightArmX: 0,
          leftForearmX: -1.5 - (isKnocking ? knockProgress * 0.3 : 0),
          rightForearmX: -0.2,
          leftHandZ: isKnocking ? knockProgress * 0.3 : 0,
          rightHandZ: 0,
          headX: 0.1, headY: 0, headZ: 0,
          neckY: 0,
          chestY: Math.sin(nav.knockPhase * 2) * 0.005,
          spineX: 0.05,
          spineZ: 0,
        };
        
        if (time > nav.stateTimer - 0.5) {
          nav.currentState = 'idle';
        }
        break;
      }

      case 'idle':
      default: {
        const breathTime = time + nav.breathOffset;
        const breathCycle = Math.sin(breathTime * 0.8);
        const microMove = Math.sin(time * 0.3) * 0.01;
        
        targetPose = {
          leftArmZ: 0.05 + microMove, 
          rightArmZ: -0.05 - microMove,
          leftArmX: Math.sin(time * 0.4) * 0.015,
          rightArmX: Math.sin(time * 0.4 + 1) * 0.015,
          leftForearmX: -0.15 + breathCycle * 0.02, 
          rightForearmX: -0.15 + breathCycle * 0.02,
          leftHandZ: Math.sin(time * 0.5) * 0.015,
          rightHandZ: Math.sin(time * 0.5 + 1) * 0.015,
          headX: Math.sin(time * 0.35) * 0.02,
          headY: Math.sin(time * 0.25) * 0.03,
          headZ: Math.sin(time * 0.2) * 0.01,
          neckY: Math.sin(time * 0.3) * 0.025,
          chestY: breathCycle * 0.005,
          spineX: 0, spineZ: 0,
        };

        const leftLeg = getBone('leftUpperLeg');
        const rightLeg = getBone('rightUpperLeg');
        const leftKnee = getBone('leftLowerLeg');
        const rightKnee = getBone('rightLowerLeg');
        
        if (leftLeg) leftLeg.rotation.x = THREE.MathUtils.lerp(leftLeg.rotation.x, 0, delta * 3);
        if (rightLeg) rightLeg.rotation.x = THREE.MathUtils.lerp(rightLeg.rotation.x, 0, delta * 3);
        if (leftKnee) leftKnee.rotation.x = THREE.MathUtils.lerp(leftKnee.rotation.x, 0.05, delta * 3);
        if (rightKnee) rightKnee.rotation.x = THREE.MathUtils.lerp(rightKnee.rotation.x, 0.05, delta * 3);
        break;
      }
    }

    // Smooth interpolation
    const lerp = (c: number, t: number, s: number) => c + (t - c) * s;
    const smoothSpeed = delta * 5;

    currentPoseRef.current = {
      leftArmZ: lerp(currentPoseRef.current.leftArmZ, targetPose.leftArmZ ?? 0.05, smoothSpeed),
      leftArmX: lerp(currentPoseRef.current.leftArmX, targetPose.leftArmX ?? 0, smoothSpeed),
      rightArmZ: lerp(currentPoseRef.current.rightArmZ, targetPose.rightArmZ ?? -0.05, smoothSpeed),
      rightArmX: lerp(currentPoseRef.current.rightArmX, targetPose.rightArmX ?? 0, smoothSpeed),
      leftForearmX: lerp(currentPoseRef.current.leftForearmX, targetPose.leftForearmX ?? -0.15, smoothSpeed),
      rightForearmX: lerp(currentPoseRef.current.rightForearmX, targetPose.rightForearmX ?? -0.15, smoothSpeed),
      leftHandZ: lerp(currentPoseRef.current.leftHandZ, targetPose.leftHandZ ?? 0, smoothSpeed),
      rightHandZ: lerp(currentPoseRef.current.rightHandZ, targetPose.rightHandZ ?? 0, smoothSpeed),
      headX: lerp(currentPoseRef.current.headX, targetPose.headX ?? 0, smoothSpeed),
      headY: lerp(currentPoseRef.current.headY, targetPose.headY ?? 0, smoothSpeed),
      headZ: lerp(currentPoseRef.current.headZ, targetPose.headZ ?? 0, smoothSpeed),
      neckY: lerp(currentPoseRef.current.neckY, targetPose.neckY ?? 0, smoothSpeed),
      chestY: lerp(currentPoseRef.current.chestY, targetPose.chestY ?? 0, smoothSpeed),
      spineX: lerp(currentPoseRef.current.spineX, targetPose.spineX ?? 0, smoothSpeed),
      spineZ: lerp(currentPoseRef.current.spineZ, targetPose.spineZ ?? 0, smoothSpeed),
    };

    // Apply to bones
    const leftUpperArm = getBone('leftUpperArm');
    const rightUpperArm = getBone('rightUpperArm');
    const leftLowerArm = getBone('leftLowerArm');
    const rightLowerArm = getBone('rightLowerArm');
    const leftHand = getBone('leftHand');
    const rightHand = getBone('rightHand');
    const head = getBone('head');
    const neck = getBone('neck');
    const chest = getBone('chest');
    const spine = getBone('spine');
    const hips = getBone('hips');

    if (leftUpperArm) {
      leftUpperArm.rotation.z = currentPoseRef.current.leftArmZ;
      leftUpperArm.rotation.x = currentPoseRef.current.leftArmX;
    }
    if (rightUpperArm) {
      rightUpperArm.rotation.z = currentPoseRef.current.rightArmZ;
      rightUpperArm.rotation.x = currentPoseRef.current.rightArmX;
    }
    if (leftLowerArm) leftLowerArm.rotation.x = currentPoseRef.current.leftForearmX;
    if (rightLowerArm) rightLowerArm.rotation.x = currentPoseRef.current.rightForearmX;
    if (leftHand) leftHand.rotation.z = currentPoseRef.current.leftHandZ;
    if (rightHand) rightHand.rotation.z = currentPoseRef.current.rightHandZ;
    if (head) {
      head.rotation.x = currentPoseRef.current.headX;
      head.rotation.y = currentPoseRef.current.headY;
      head.rotation.z = currentPoseRef.current.headZ;
    }
    if (neck) neck.rotation.y = currentPoseRef.current.neckY;
    if (chest) chest.position.y = currentPoseRef.current.chestY + Math.sin(time * 1.5) * 0.005;
    if (spine) {
      spine.rotation.x = currentPoseRef.current.spineX;
      spine.rotation.z = currentPoseRef.current.spineZ;
    }

    if (nav.currentState !== 'sneaking' && hips) {
      hips.position.y = THREE.MathUtils.lerp(hips.position.y, 0, delta * 3);
    }

    // === MORPH TARGET ANIMATIONS ===
    // Blink
    if (morphTargetsRef.current.has('blink') || morphTargetsRef.current.has('eyesClosed')) {
      const blinkVal = Math.sin(time * 3) > 0.98 ? 1 : 0;
      setMorphValue('blink', blinkVal);
      setMorphValue('eyesClosed', blinkVal);
    }

    // === LIP SYNC - IMPROVED ===
    if (isSpeaking) {
      const fastCycle = time * 10;
      const realAudio = audioAmplitudeRef.current;
      
      let currentAudio: number;
      
      if (realAudio < 0.05) {
        const speechPattern = Math.sin(fastCycle) * 0.5 + 0.5;
        const harmonic = Math.sin(fastCycle * 1.7) * 0.25;
        const syllableCycle = (time * 6) % 1;
        const wordCycle = (time * 2) % 1;
        const sentenceCycle = (time * 0.4) % 1;
        
        const isSyllableGap = syllableCycle > 0.8;
        const isWordGap = wordCycle > 0.88;
        const isSentencePause = sentenceCycle > 0.92;
        
        if (isSentencePause) {
          currentAudio = 0.01;
          isSilentRef.current = true;
        } else if (isWordGap) {
          currentAudio = 0.05;
          isSilentRef.current = false;
        } else if (isSyllableGap) {
          currentAudio = 0.12;
          isSilentRef.current = false;
        } else {
          currentAudio = 0.2 + (speechPattern * 0.25) + (harmonic * 0.1);
          isSilentRef.current = false;
        }
        
        silenceTimerRef.current = 0;
      } else {
        currentAudio = realAudio;
        
        if (realAudio < 0.05) {
          silenceTimerRef.current += delta;
        } else {
          silenceTimerRef.current = 0;
          isSilentRef.current = false;
        }
        
        if (silenceTimerRef.current > 0.3) {
          isSilentRef.current = true;
        }
      }
      
      resetAllMorphs();
      
      if (!isSilentRef.current && currentAudio > 0.06) {
        smoothedAmplitudeRef.current += (currentAudio - smoothedAmplitudeRef.current) * 0.25;
        const oscillation = Math.sin(fastCycle * 0.6) * 0.12;
        const mouthValue = Math.max(0, Math.min(0.5, smoothedAmplitudeRef.current * 0.6 + oscillation));
        
        setMorphValue('mouthOpen', mouthValue);
        setMorphValue('mouthopen', mouthValue);
        setMorphValue('mouth_open', mouthValue);
        setMorphValue('a', mouthValue);
        setMorphValue('aa', mouthValue);
        setMorphValue('A', mouthValue);
        setMorphValue('jawOpen', mouthValue * 0.8);
        
        setMorphValue('happy', mouthValue * 0.3);
        setMorphValue('smile', mouthValue * 0.3);
      } else {
        smoothedAmplitudeRef.current *= 0.8;
      }
    } else {
      smoothedAmplitudeRef.current = 0;
      silenceTimerRef.current = 0;
      isSilentRef.current = false;
      resetAllMorphs();
    }
  });

  if (error) {
    return (
      <group ref={groupRef}>
        <mesh>
          <boxGeometry args={[2, 2, 2]} />
          <meshStandardMaterial color="red" />
        </mesh>
      </group>
    );
  }

  if (!isLoaded) {
    return (
      <group ref={groupRef}>
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={primaryColor} wireframe />
        </mesh>
      </group>
    );
  }

  return <group ref={groupRef} scale={[1.5, 1.5, 1.5]} rotation={[0, Math.PI, 0]} />;
}
