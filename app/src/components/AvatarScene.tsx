import { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import { useStore } from "@/store";

const vertexShader = `
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
    vec3 ns = n_ * D.wyz - D.xzx;
    
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
    
    float noise = snoise(position * 2.0 + uTime * 0.5);
    float listeningBoost = uIsListening * 0.3 * sin(uTime * 3.0);
    float speakingBoost = uIsSpeaking * 0.5 * sin(uTime * 8.0);
    
    float displacement = (noise * 0.3 + listeningBoost + speakingBoost) * uComplexity;
    vDisplacement = displacement;
    
    vec3 newPosition = position + normal * displacement;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

const fragmentShader = `
  uniform vec3 uPrimaryColor;
  uniform vec3 uSecondaryColor;
  uniform float uTime;
  uniform float uIsListening;
  uniform float uIsSpeaking;
  
  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vDisplacement;
  
  void main() {
    vec3 color = mix(uPrimaryColor, uSecondaryColor, vDisplacement + 0.5);
    
    float glow = uIsListening * 0.3 + uIsSpeaking * 0.5;
    color += vec3(glow * 0.5);
    
    vec3 viewDirection = normalize(cameraPosition - vNormal);
    float fresnel = pow(1.0 - abs(dot(viewDirection, vNormal)), 2.0);
    color += uSecondaryColor * fresnel * 0.5;
    
    gl_FragColor = vec4(color, 0.95);
  }
`;

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
  
  // Create colors from hex strings
  const primaryColorObj = useMemo(() => new THREE.Color(primaryColor), [primaryColor]);
  const secondaryColorObj = useMemo(() => new THREE.Color(secondaryColor), [secondaryColor]);
  
  // Initialize uniforms once
  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uPrimaryColor: { value: primaryColorObj },
    uSecondaryColor: { value: secondaryColorObj },
    uIsListening: { value: 0 },
    uIsSpeaking: { value: 0 },
    uComplexity: { value: complexity },
  }), []); // Empty deps - we update values manually

  // Update uniform values when props change
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uPrimaryColor.value.set(primaryColor);
      materialRef.current.uniforms.uSecondaryColor.value.set(secondaryColor);
      materialRef.current.uniforms.uComplexity.value = complexity;
      console.log('Avatar colors updated:', { primaryColor, secondaryColor });
    }
  }, [primaryColor, secondaryColor, complexity]);

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      materialRef.current.uniforms.uIsListening.value = isListening ? 1 : 0;
      materialRef.current.uniforms.uIsSpeaking.value = isSpeaking ? 1 : 0;
    }
    
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.002;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[2, 128, 128]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  );
}

function ParticleField({ color }: { color: string }) {
  const pointsRef = useRef<THREE.Points>(null);
  const particleCount = 200;
  const materialRef = useRef<THREE.PointsMaterial>(null);
  
  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const vel = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      pos[i3] = (Math.random() - 0.5) * 15;
      pos[i3 + 1] = (Math.random() - 0.5) * 15;
      pos[i3 + 2] = (Math.random() - 0.5) * 15;
      
      vel[i3] = (Math.random() - 0.5) * 0.02;
      vel[i3 + 1] = (Math.random() - 0.5) * 0.02;
      vel[i3 + 2] = (Math.random() - 0.5) * 0.02;
    }
    
    return [pos, vel];
  }, []);

  // Update particle color when it changes
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.color.set(color);
    }
  }, [color]);

  useFrame(() => {
    if (pointsRef.current) {
      const posArray = pointsRef.current.geometry.attributes.position.array as Float32Array;
      
      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        posArray[i3] += velocities[i3];
        posArray[i3 + 1] += velocities[i3 + 1];
        posArray[i3 + 2] += velocities[i3 + 2];
        
        if (Math.abs(posArray[i3]) > 7.5) velocities[i3] *= -1;
        if (Math.abs(posArray[i3 + 1]) > 7.5) velocities[i3 + 1] *= -1;
        if (Math.abs(posArray[i3 + 2]) > 7.5) velocities[i3 + 2] *= -1;
      }
      
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
      pointsRef.current.rotation.y += 0.001;
    }
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
        ref={materialRef}
        size={0.05}
        color={color}
        transparent
        opacity={0.6}
        sizeAttenuation
      />
    </points>
  );
}

function Scene() {
  const { currentPersona, isListening, isSpeaking } = useStore();
  
  // Get avatar config with defaults
  const avatarConfig = currentPersona?.avatar_config || {
    primary_color: "#6366f1",
    secondary_color: "#8b5cf6",
    glow_color: "#a78bfa",
    animation_style: "flowing",
    complexity: 0.7,
  };

  // Force re-render when persona changes
  const personaKey = currentPersona?.id || 'default';

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight 
        key={`light1-${personaKey}`}
        position={[10, 10, 10]} 
        intensity={1} 
        color={avatarConfig.primary_color} 
      />
      <pointLight 
        key={`light2-${personaKey}`}
        position={[-10, -10, -10]} 
        intensity={0.5} 
        color={avatarConfig.secondary_color} 
      />
      
      <AnimatedSphere
        key={`sphere-${personaKey}`}
        primaryColor={avatarConfig.primary_color}
        secondaryColor={avatarConfig.secondary_color}
        isListening={isListening}
        isSpeaking={isSpeaking}
        complexity={avatarConfig.complexity}
      />
      
      <ParticleField 
        key={`particles-${personaKey}`}
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
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.5}
      />
    </>
  );
}

export function AvatarScene() {
  const { currentPersona } = useStore();
  const personaKey = currentPersona?.id || 'default';

  return (
    <div className="w-full h-full bg-gradient-to-b from-background via-background to-background/50">
      <Canvas 
        key={personaKey}
        camera={{ position: [0, 0, 8], fov: 45 }}
      >
        <Scene />
      </Canvas>
      
      <div className="absolute bottom-4 left-4 text-xs text-muted-foreground bg-card/50 backdrop-blur-sm px-3 py-2 rounded-lg">
        <p>Drag to rotate • Scroll to zoom</p>
      </div>
    </div>
  );
}
