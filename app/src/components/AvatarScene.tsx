import { useRef, useMemo, useEffect, useState, memo } from "react";
import { Canvas, useFrame, extend } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import { useStore } from "@/store";
import { VrmAvatar, BASE_VRMA_PATHS } from "./VrmAvatar";
import { GenericAvatar } from "./GenericAvatar";
import { loadVrmAsBlobUrl, loadModelAsBlobUrl } from "@/services/vrmLibrary";
import { EmoteMenu } from "./EmoteMenu";
import { toast } from "sonner";

// Extend THREE namespace for R3F v9 compatibility
// This ensures geometry elements like circleGeometry are recognized
// Keys must match the lowercase JSX element names that drei uses
extend({
  circleGeometry: THREE.CircleGeometry,
  ringGeometry: THREE.RingGeometry,
  planeGeometry: THREE.PlaneGeometry,
  boxGeometry: THREE.BoxGeometry,
  sphereGeometry: THREE.SphereGeometry,
  cylinderGeometry: THREE.CylinderGeometry,
  coneGeometry: THREE.ConeGeometry,
  torusGeometry: THREE.TorusGeometry,
  tubeGeometry: THREE.TubeGeometry,
  extrudeGeometry: THREE.ExtrudeGeometry,
  latheGeometry: THREE.LatheGeometry,
  shapeGeometry: THREE.ShapeGeometry,
});

// ============================================
// SHADERS
// ============================================

const sphereVertexShader = `
  uniform float uTime;
  uniform float uIsListening;
  uniform float uIsSpeaking;
  uniform float uComplexity;
  
  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vDisplacement;
  
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    
    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    
    float n_ = 0.142857142857;
    vec3  ns = n_ * D.wyz - D.xzx;
    
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }
  
  void main() {
    vUv = uv;
    vNormal = normal;
    
    float noise = snoise(position * uComplexity + uTime * 0.5);
    
    float listeningEffect = uIsListening * sin(uTime * 10.0) * 0.1;
    float speakingEffect = uIsSpeaking * sin(uTime * 20.0) * 0.15;
    
    vDisplacement = noise + listeningEffect + speakingEffect;
    
    vec3 newPosition = position + normal * vDisplacement * 0.3;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

const sphereFragmentShader = `
  uniform vec3 uPrimaryColor;
  uniform vec3 uSecondaryColor;
  uniform float uTime;
  uniform float uIsListening;
  uniform float uIsSpeaking;
  
  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vDisplacement;
  
  void main() {
    vec3 color = mix(uPrimaryColor, uSecondaryColor, vUv.y + vDisplacement * 0.5);
    
    float glow = 0.5 + 0.5 * sin(uTime * 2.0);
    color += vec3(glow * 0.2) * uIsListening;
    color += vec3(glow * 0.3) * uIsSpeaking;
    
    float fresnel = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
    color += uSecondaryColor * fresnel * 0.5;
    
    gl_FragColor = vec4(color, 0.9);
  }
`;

// ============================================
// COMPONENTS
// ============================================

function AnimatedSphere({ 
  primaryColor, 
  secondaryColor, 
  isListening, 
  isSpeaking, 
  complexity 
}: { 
  primaryColor: string; 
  secondaryColor: string; 
  isListening: boolean;
  isSpeaking: boolean;
  complexity: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uPrimaryColor: { value: new THREE.Color(primaryColor) },
    uSecondaryColor: { value: new THREE.Color(secondaryColor) },
    uIsListening: { value: 0 },
    uIsSpeaking: { value: 0 },
    uComplexity: { value: complexity },
  }), []); // Empty deps - we update colors via useEffect for real-time changes

  // Update colors in real-time when they change
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uPrimaryColor.value.set(primaryColor);
      materialRef.current.uniforms.uSecondaryColor.value.set(secondaryColor);
    }
  }, [primaryColor, secondaryColor]);

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      materialRef.current.uniforms.uIsListening.value = isListening ? 1 : 0;
      materialRef.current.uniforms.uIsSpeaking.value = isSpeaking ? 1 : 0;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.5, 128, 128]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={sphereVertexShader}
        fragmentShader={sphereFragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  );
}

function ParticleField({ color }: { color: string }) {
  const pointsRef = useRef<THREE.Points>(null);
  const particleCount = 200;

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const vel = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      pos[i3] = (Math.random() - 0.5) * 10;
      pos[i3 + 1] = (Math.random() - 0.5) * 10;
      pos[i3 + 2] = (Math.random() - 0.5) * 10;
      
      vel[i3] = (Math.random() - 0.5) * 0.01;
      vel[i3 + 1] = (Math.random() - 0.5) * 0.01;
      vel[i3 + 2] = (Math.random() - 0.5) * 0.01;
    }
    
    return [pos, vel];
  }, []);

  useFrame(() => {
    if (!pointsRef.current) return;
    
    const positionArray = pointsRef.current.geometry.attributes.position.array as Float32Array;
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positionArray[i3] += velocities[i3];
      positionArray[i3 + 1] += velocities[i3 + 1];
      positionArray[i3 + 2] += velocities[i3 + 2];
      
      if (Math.abs(positionArray[i3]) > 5) velocities[i3] *= -1;
      if (Math.abs(positionArray[i3 + 1]) > 5) velocities[i3 + 1] *= -1;
      if (Math.abs(positionArray[i3 + 2]) > 5) velocities[i3 + 2] *= -1;
    }
    
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    pointsRef.current.rotation.y += 0.001;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color={color}
        transparent
        opacity={0.6}
        sizeAttenuation
      />
    </points>
  );
}

// VRM Wrapper Component - Memoized to prevent reloads
const VrmAvatarWrapper = memo(function VrmAvatarWrapper({ 
  modelId, 
  isListening, 
  isSpeaking, 
  primaryColor,
  vrmaPaths,
  enabledBaseVrmas,
  autoAnimation,
}: { 
  modelId: string | undefined;
  isListening: boolean;
  isSpeaking: boolean;
  primaryColor: string;
  vrmaPaths?: Record<string, string>;
  enabledBaseVrmas: string[];
  autoAnimation: boolean;
}) {
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!modelId) {
      setIsLoading(false);
      return;
    }
    
    // Handle bundled assets (bundled://default/avatar.vrm)
    if (modelId.startsWith('bundled://')) {
      const bundledPath = modelId.replace('bundled://', '/personas/');
      console.log('[VrmAvatarWrapper] Loading bundled VRM:', bundledPath);
      setModelUrl(bundledPath);
      setIsLoading(false);
      return;
    }
    
    // Only reload if modelId changes
    setIsLoading(true);
    loadVrmAsBlobUrl(modelId)
      .then(url => {
        setModelUrl(url);
        setIsLoading(false);
      })
      .catch(err => {
        console.error("[VrmAvatarWrapper] Failed to load VRM:", err);
        setIsLoading(false);
      });
  }, [modelId]);

  if (isLoading) {
    return (
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshStandardMaterial color={primaryColor} wireframe />
      </mesh>
    );
  }

  if (!modelUrl) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="red" transparent opacity={0.5} />
      </mesh>
    );
  }

  return (
    <VrmAvatar
      modelUrl={modelUrl}
      isListening={isListening}
      isSpeaking={isSpeaking}
      primaryColor={primaryColor}
      personaVrmas={vrmaPaths}
      enabledBaseVrmas={enabledBaseVrmas}
      autoAnimationEnabled={autoAnimation}
    />
  );
});

// GLB Wrapper Component
function GenericAvatarWrapper({ 
  modelId, 
  isListening, 
  isSpeaking, 
  primaryColor 
}: { 
  modelId: string | undefined;
  isListening: boolean;
  isSpeaking: boolean;
  primaryColor: string;
}) {
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!modelId) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    loadModelAsBlobUrl(modelId)
      .then(url => {
        setModelUrl(url);
        setIsLoading(false);
      })
      .catch(err => {
        console.error("[GenericAvatarWrapper] Failed to load model:", err);
        setIsLoading(false);
      });
  }, [modelId]);

  if (isLoading) {
    return (
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshStandardMaterial color={primaryColor} wireframe />
      </mesh>
    );
  }

  if (!modelUrl) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="red" transparent opacity={0.5} />
      </mesh>
    );
  }

  return (
    <GenericAvatar
      modelUrl={modelUrl}
      isListening={isListening}
      isSpeaking={isSpeaking}
      primaryColor={primaryColor}
    />
  );
}

// ============================================
// SCENE
// ============================================

function Scene() {
  const { currentPersona, isListening, isSpeaking } = useStore();
  
  // Get avatar config with defaults
  const avatarConfig = currentPersona?.avatar_config || {
    type: "abstract" as const,
    primary_color: "#6366f1",
    secondary_color: "#8b5cf6",
    glow_color: "#a78bfa",
    shape_type: "sphere",
    animation_style: "flowing",
    complexity: 0.7,
  };

  // Animation settings from persona config - memoized to prevent re-renders
  const enabledBaseVrmas = useMemo(() => 
    (avatarConfig as any).enabled_base_vrmas || Object.keys(BASE_VRMA_PATHS), 
    [(avatarConfig as any).enabled_base_vrmas]
  );
  const autoAnimation = (avatarConfig as any).auto_animation !== false; // default true
  const vrmaPaths = useMemo(() => 
    (avatarConfig as any).vrma_paths || {}, 
    [(avatarConfig as any).vrma_paths]
  );

  const avatarType = avatarConfig.type || "abstract";



  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight 
        position={[10, 10, 10]} 
        intensity={1} 
        color={avatarConfig.primary_color} 
      />
      <pointLight 
        position={[-10, -10, -10]} 
        intensity={0.5} 
        color={avatarConfig.secondary_color} 
      />
      
      {avatarType === "abstract" && (
        <AnimatedSphere
          primaryColor={avatarConfig.primary_color}
          secondaryColor={avatarConfig.secondary_color}
          isListening={isListening}
          isSpeaking={isSpeaking}
          complexity={avatarConfig.complexity}
        />
      )}
      
      {avatarType === "vrm" && (
        <VrmAvatarWrapper
          modelId={(avatarConfig as any).vrm_id}
          isListening={isListening}
          isSpeaking={isSpeaking}
          primaryColor={avatarConfig.primary_color}
          vrmaPaths={vrmaPaths}
          enabledBaseVrmas={enabledBaseVrmas}
          autoAnimation={autoAnimation}
        />
      )}
      
      {(avatarType === "glb" || avatarType === "gltf") && (
        <GenericAvatarWrapper
          modelId={(avatarConfig as any).model_id}
          isListening={isListening}
          isSpeaking={isSpeaking}
          primaryColor={avatarConfig.primary_color}
        />
      )}
      
      <ParticleField 
        color={avatarConfig.glow_color} 
      />
      
      <Stars
        radius={50}
        depth={50}
        count={1000}
        factor={4}
        saturation={0.5}
        fade
      />
      
      <OrbitControls
        enableZoom={true}
        enablePan={true}
        autoRotate={false}
        minPolarAngle={0}
        maxPolarAngle={Math.PI}
        minDistance={0.3}
        maxDistance={20}
        enableDamping
        dampingFactor={0.05}
      />
    </>
  );
}

// Sound effect for glass knocking
const playKnockSound = () => {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(200, audioContext.currentTime + 0.05);
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.05);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.05);
  } catch (e) {
    console.error('Failed to play knock sound:', e);
  }
};

export function AvatarScene() {
  const { currentPersona, updatePersona } = useStore();
  
  // Get avatar config
  const avatarConfig = currentPersona?.avatar_config || {
    type: "abstract" as const,
    primary_color: "#6366f1",
  };
  
  const isVrm = avatarConfig.type === "vrm";
  const vrmaPaths = (avatarConfig as any).vrma_paths || {};
  const enabledBaseVrmas = (avatarConfig as any).enabled_base_vrmas || Object.keys(BASE_VRMA_PATHS).filter(k => k !== 'idle');
  const autoAnimation = (avatarConfig as any).auto_animation !== false;
  
  useEffect(() => {
    const handleKnock = () => playKnockSound();
    window.addEventListener('avatar-knock', handleKnock);
    return () => window.removeEventListener('avatar-knock', handleKnock);
  }, []);

  const handlePlayEmote = (name: string) => {
    if ((window as any).vrmPlayEmote) {
      (window as any).vrmPlayEmote(name);
    }
  };

  const handleToggleBaseVrma = (name: string, enabled: boolean) => {
    if (!currentPersona) return;
    const newEnabled = enabled 
      ? [...enabledBaseVrmas, name]
      : enabledBaseVrmas.filter((n: string) => n !== name);
    const updated: typeof currentPersona = {
      ...currentPersona,
      avatar_config: { 
        ...avatarConfig, 
        enabled_base_vrmas: newEnabled 
      } as any,
      updated_at: new Date().toISOString(),
    };
    updatePersona(updated);
  };

  const handleToggleAutoAnimation = (enabled: boolean) => {
    if (!currentPersona) return;
    const updated: typeof currentPersona = {
      ...currentPersona,
      avatar_config: { 
        ...avatarConfig, 
        auto_animation: enabled 
      } as any,
      updated_at: new Date().toISOString(),
    };
    updatePersona(updated);
    toast.success(enabled ? "Auto animation enabled" : "Auto animation disabled");
  };

  return (
    <div className="w-full h-full bg-gradient-to-b from-background via-background to-background/50 relative">
      {/* Emote Menu - DOM overlay, locked to corner */}
      {isVrm && (
        <div className="absolute top-4 left-4 z-20">
          <EmoteMenu
            baseVrmas={BASE_VRMA_PATHS}
            personaVrmas={vrmaPaths}
            enabledBaseVrmas={enabledBaseVrmas}
            autoAnimation={autoAnimation}
            onToggleBaseVrma={handleToggleBaseVrma}
            onToggleAutoAnimation={handleToggleAutoAnimation}
            onPlayEmote={handlePlayEmote}
          />
        </div>
      )}
      
      <Canvas 
        camera={{ position: [0, 1.5, 6], fov: 50 }}
      >
        <Scene />
      </Canvas>
      
      <div className="absolute bottom-4 left-4 text-xs text-muted-foreground bg-card/50 backdrop-blur-sm px-3 py-2 rounded-lg">
        <p>Left drag: Rotate • Scroll: Zoom • Shift/Right-click drag: Move</p>
      </div>
    </div>
  );
}
