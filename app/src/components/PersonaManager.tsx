import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Plus, 
  Edit2, 
  Trash2, 
  Bot, 
  Mic, 
  Sparkles,
  Palette,
  Save,
  X,
  Check,
  Wand2,
  Volume2,
  Upload,
  Trash,
  Edit3,
  FileBox,
  Loader2
} from "lucide-react";
import { useStore } from "@/store";
import type { Persona, AvatarConfig } from "@/types";
// @ts-ignore - Used in async function but TypeScript doesn't detect it
import { memoryToolsService } from "@/services/memoryTools";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Textarea } from "./ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "./ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ollamaService } from "@/services/ollama";
import { 
  listVrmLibrary, 
  saveVrmToLibrary, 
  deleteVrmFromLibrary, 
  renameVrmInLibrary,
  formatFileSize,
  type VrmEntry 
} from "@/services/vrmLibrary";
import {
  listPersonaVrmas,
  saveVrmaToPersona,
  deleteVrma,
  type VrmaEntry
} from "@/services/vrmaLibrary";

// KittenTTS Voices - 8 selectable AI voices
const KITTEN_VOICES = [
  { value: "Bella", label: "Bella (Female)", gender: "female" },
  { value: "Luna", label: "Luna (Female)", gender: "female" },
  { value: "Rosie", label: "Rosie (Female)", gender: "female" },
  { value: "Kiki", label: "Kiki (Female)", gender: "female" },
  { value: "Jasper", label: "Jasper (Male)", gender: "male" },
  { value: "Bruno", label: "Bruno (Male)", gender: "male" },
  { value: "Hugo", label: "Hugo (Male)", gender: "male" },
  { value: "Leo", label: "Leo (Male)", gender: "male" },
] as const;

// Voice groups for organized display
const VOICE_GROUPS = {
  female: KITTEN_VOICES.filter(v => v.gender === "female"),
  male: KITTEN_VOICES.filter(v => v.gender === "male"),
};

interface VrmLibrarySelectorProps {
  selectedVrmId?: string;
  onSelect: (vrmId: string | undefined, modelUrl: string | undefined) => void;
}

function VrmLibrarySelector({ selectedVrmId, onSelect }: VrmLibrarySelectorProps) {
  const [vrms, setVrms] = useState<VrmEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load VRM library on mount
  useEffect(() => {
    loadLibrary();
  }, []);

  const loadLibrary = async () => {
    try {
      setIsLoading(true);
      const library = await listVrmLibrary();
      setVrms(library.entries);
    } catch (error) {
      console.error("Failed to load VRM library:", error);
      toast.error("Failed to load VRM library");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.vrm')) {
      toast.error("Please select a .vrm file");
      return;
    }

    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      toast.error("File too large. Maximum size is 100MB");
      return;
    }

    try {
      setIsUploading(true);
      toast.info(`Uploading ${file.name}...`);
      
      const name = file.name.replace(/\.vrm$/i, "");
      const entry = await saveVrmToLibrary(name, file);
      
      toast.success(`"${entry.name}" added to library`);
      await loadLibrary();
      
      // Auto-select the newly uploaded VRM
      onSelect(entry.id, undefined);
    } catch (error) {
      console.error("Failed to upload VRM:", error);
      toast.error("Failed to upload VRM file");
    } finally {
      setIsUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDelete = async (vrm: VrmEntry, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!confirm(`Delete "${vrm.name}" from library?`)) return;

    try {
      await deleteVrmFromLibrary(vrm.id);
      toast.success(`"${vrm.name}" deleted`);
      
      // If this was the selected VRM, clear selection
      if (selectedVrmId === vrm.id) {
        onSelect(undefined, undefined);
      }
      
      await loadLibrary();
    } catch (error) {
      console.error("Failed to delete VRM:", error);
      toast.error("Failed to delete VRM");
    }
  };

  const startEditing = (vrm: VrmEntry, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(vrm.id);
    setEditingName(vrm.name);
  };

  const saveRename = async (vrmId: string) => {
    if (!editingName.trim()) {
      setEditingId(null);
      return;
    }

    try {
      await renameVrmInLibrary(vrmId, editingName.trim());
      toast.success("Renamed successfully");
      await loadLibrary();
    } catch (error) {
      console.error("Failed to rename VRM:", error);
      toast.error("Failed to rename");
    } finally {
      setEditingId(null);
    }
  };

  const selectVrm = (vrm: VrmEntry) => {
    // Just pass the vrmId - AvatarScene will load the blob URL fresh
    onSelect(vrm.id, undefined);
    toast.success(`Selected "${vrm.name}"`);
  };

  return (
    <div className="space-y-3 pt-3 border-t">
      <div className="flex items-center justify-between">
        <Label className="text-xs font-semibold">VRM Library</Label>
        <input
          type="file"
          accept=".vrm"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
        >
          {isUploading ? (
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
          ) : (
            <Upload className="w-3 h-3 mr-1" />
          )}
          {isUploading ? "Uploading..." : "Upload VRM"}
        </Button>
      </div>

      {isLoading ? (
        <div className="text-center py-4 text-muted-foreground">
          <Loader2 className="w-5 h-5 animate-spin mx-auto mb-2" />
          Loading library...
        </div>
      ) : vrms.length === 0 ? (
        <div className="text-center py-6 border rounded-lg bg-muted/50">
          <FileBox className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">No VRM files in library</p>
          <p className="text-xs text-muted-foreground mt-1">
            Upload a .vrm file to use it as your avatar
          </p>
        </div>
      ) : (
        <div className="space-y-1 max-h-48 overflow-auto border rounded-lg p-1">
          {vrms.map((vrm) => (
            <div
              key={vrm.id}
              onClick={() => selectVrm(vrm)}
              className={cn(
                "flex items-center gap-2 p-2 rounded cursor-pointer transition-colors",
                selectedVrmId === vrm.id
                  ? "bg-primary/10 border border-primary/30"
                  : "hover:bg-muted border border-transparent"
              )}
            >
              <div className="w-8 h-8 rounded bg-muted flex items-center justify-center flex-shrink-0">
                <FileBox className="w-4 h-4 text-muted-foreground" />
              </div>
              
              <div className="flex-1 min-w-0">
                {editingId === vrm.id ? (
                  <Input
                    value={editingName}
                    onChange={(e) => setEditingName(e.target.value)}
                    onBlur={() => saveRename(vrm.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") saveRename(vrm.id);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    className="h-6 text-xs py-0"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <p className="text-sm font-medium truncate">{vrm.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(vrm.size_bytes)}
                    </p>
                  </>
                )}
              </div>

              <div className="flex items-center gap-1">
                {selectedVrmId === vrm.id && (
                  <Check className="w-4 h-4 text-primary" />
                )}
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={(e) => startEditing(vrm, e)}
                >
                  <Edit3 className="w-3 h-3" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-destructive hover:text-destructive"
                  onClick={(e) => handleDelete(vrm, e)}
                >
                  <Trash className="w-3 h-3" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedVrmId && (
        <p className="text-xs text-muted-foreground">
          Selected VRM will be used for this persona. Supports lip sync during speech!
        </p>
      )}
    </div>
  );
}

// VRMA Animation Library Selector
interface VrmaLibrarySelectorProps {
  personaId?: string;
  vrmaPaths?: Record<string, string>;
  onChange: (vrmaPaths: Record<string, string>) => void;
}

function VrmaLibrarySelector({ personaId, vrmaPaths = {}, onChange }: VrmaLibrarySelectorProps) {
  const [vrmas, setVrmas] = useState<VrmaEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (personaId) {
      loadVrmas();
    }
  }, [personaId]);

  const loadVrmas = async () => {
    if (!personaId) return;
    try {
      setIsLoading(true);
      const entries = await listPersonaVrmas(personaId);
      setVrmas(entries);
    } catch (error) {
      console.error("Failed to load VRMAs:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !personaId) return;

    if (!file.name.toLowerCase().endsWith('.vrma')) {
      toast.error("Please select a .vrma file");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      toast.error("File too large. Maximum size is 50MB");
      return;
    }

    try {
      setIsUploading(true);
      toast.info(`Uploading ${file.name}...`);
      
      const entry = await saveVrmaToPersona(personaId, file);
      
      toast.success(`"${entry.name}" added`);
      await loadVrmas();
      
      // Update parent with new paths
      onChange({ ...vrmaPaths, [entry.name]: entry.path });
    } catch (error) {
      console.error("Failed to upload VRMA:", error);
      toast.error("Failed to upload VRMA file");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDelete = async (vrma: VrmaEntry) => {
    if (!personaId || !confirm(`Delete "${vrma.name}"?`)) return;

    try {
      await deleteVrma(personaId, vrma.id);
      toast.success(`"${vrma.name}" deleted`);
      
      // Remove from paths
      const newPaths = { ...vrmaPaths };
      delete newPaths[vrma.name];
      onChange(newPaths);
      
      await loadVrmas();
    } catch (error) {
      console.error("Failed to delete VRMA:", error);
      toast.error("Failed to delete VRMA");
    }
  };

  return (
    <div className="space-y-2 pt-2 border-t">
      <div className="flex items-center justify-between">
        <Label className="text-xs font-semibold">Custom Animations (VRMA)</Label>
        <input
          type="file"
          accept=".vrma"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading || !personaId}
        >
          {isUploading ? (
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
          ) : (
            <Upload className="w-3 h-3 mr-1" />
          )}
          {isUploading ? "Uploading..." : "Upload VRMA"}
        </Button>
      </div>

      {!personaId && (
        <p className="text-xs text-muted-foreground">
          Save the persona first to upload custom animations
        </p>
      )}

      {isLoading ? (
        <div className="text-center py-2">
          <Loader2 className="w-4 h-4 animate-spin mx-auto" />
        </div>
      ) : vrmas.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No custom animations. Upload .vrma files to add unique animations for this persona.
        </p>
      ) : (
        <div className="space-y-1 max-h-32 overflow-auto border rounded p-1">
          {vrmas.map((vrma) => (
            <div
              key={vrma.id}
              className="flex items-center justify-between px-2 py-1.5 rounded text-sm hover:bg-muted/50"
            >
              <div className="flex items-center gap-2">
                <Sparkles className="w-3 h-3 text-muted-foreground" />
                <span className="truncate max-w-[120px]">{vrma.name}</span>
                <span className="text-xs text-muted-foreground">
                  {formatFileSize(vrma.size)}
                </span>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-5 w-5 text-destructive hover:text-destructive"
                onClick={() => handleDelete(vrma)}
              >
                <Trash className="w-3 h-3" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Base Animation Selector
interface BaseAnimationSelectorProps {
  enabledVrmas?: string[];
  onChange: (enabledVrmas: string[]) => void;
}

const BASE_ANIMATIONS = [
  { id: 'greeting', label: 'Greeting', description: 'Wave/hello animation' },
  { id: 'peaceSign', label: 'Peace Sign', description: 'Peace sign pose' },
  { id: 'shoot', label: 'Shoot', description: 'Point and shoot gesture' },
  { id: 'spin', label: 'Spin', description: '360 degree spin' },
  { id: 'modelPose', label: 'Model Pose', description: 'Fashion model pose' },
  { id: 'squat', label: 'Squat', description: 'Squat down' },
];

function BaseAnimationSelector({ enabledVrmas, onChange }: BaseAnimationSelectorProps) {
  const enabled = enabledVrmas || BASE_ANIMATIONS.map(a => a.id);
  
  const toggleAnimation = (id: string) => {
    if (enabled.includes(id)) {
      // Don't allow disabling if it would leave less than 2 animations
      if (enabled.length <= 2) {
        toast.error("Keep at least 2 animations enabled");
        return;
      }
      onChange(enabled.filter(e => e !== id));
    } else {
      onChange([...enabled, id]);
    }
  };
  
  return (
    <div className="space-y-3 pt-3 border-t">
      <div className="flex items-center justify-between">
        <Label className="text-xs font-semibold">Base Animations</Label>
        <span className="text-xs text-muted-foreground">
          {enabled.length} enabled
        </span>
      </div>
      
      <div className="space-y-1.5 max-h-40 overflow-y-auto border rounded p-2">
        {BASE_ANIMATIONS.map((anim) => (
          <div
            key={anim.id}
            className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50 cursor-pointer"
            onClick={() => toggleAnimation(anim.id)}
          >
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium">{anim.label}</div>
              <div className="text-xs text-muted-foreground truncate">
                {anim.description}
              </div>
            </div>
            <div className="ml-2">
              {enabled.includes(anim.id) ? (
                <Check className="w-4 h-4 text-green-500" />
              ) : (
                <div className="w-4 h-4 rounded-sm border-2 border-muted-foreground/30" />
              )}
            </div>
          </div>
        ))}
      </div>
      
      <p className="text-xs text-muted-foreground">
        Select which base animations are available for this persona. Changes apply immediately.
      </p>
    </div>
  );
}

interface PersonaFormProps {
  persona?: Persona;
  onSave: (persona: Partial<Persona>) => void;
  onCancel: () => void;
}

function PersonaForm({ persona, onSave, onCancel }: PersonaFormProps) {
  const { settings } = useStore();
  // Parse arrays to comma-separated strings for the form
  const wakeWordsStr = persona?.wake_words?.join(", ") || "Mimic";
  const responseWordsStr = persona?.response_words?.join(", ") || "Yes?, I'm listening, Mimic here";
  
  const [formData, setFormData] = useState({
    name: persona?.name || "",
    description: persona?.description || "",
    personality_prompt: persona?.personality_prompt || "",
    wake_words: wakeWordsStr,
    response_words: responseWordsStr,
    voice_id: persona?.voice_id || "Bella",
    avatar_config: persona?.avatar_config || {
      type: "abstract",
      primary_color: "#6366f1",
      secondary_color: "#8b5cf6",
      glow_color: "#a78bfa",
      shape_type: "sphere",
      animation_style: "flowing",
      complexity: 0.7,
    },
  });
  const [isGeneratingAvatar, setIsGeneratingAvatar] = useState(false);
  const hasCreatedVoice = !!persona?.voice_create;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Convert comma-separated strings to arrays
    const dataToSave = {
      ...formData,
      wake_words: formData.wake_words.split(",").map(s => s.trim()).filter(s => s),
      response_words: formData.response_words.split(",").map(s => s.trim()).filter(s => s),
    };
    onSave(dataToSave);
  };

  const generateAvatar = async () => {
    if (!formData.name || !formData.personality_prompt) {
      toast.error("Please provide a name and personality first");
      return;
    }

    setIsGeneratingAvatar(true);
    toast.info("Asking AI to design its avatar...");

    try {
      const mockPersona: Persona = {
        id: "temp",
        name: formData.name,
        description: formData.description,
        personality_prompt: formData.personality_prompt,
        wake_words: formData.wake_words.split(",").map(s => s.trim()).filter(s => s),
        response_words: formData.response_words.split(",").map(s => s.trim()).filter(s => s),
        voice_id: formData.voice_id,
        avatar_config: formData.avatar_config as AvatarConfig,
        memory: {
          short_term: [],
          long_term: [],
          summary: "",
          last_summarized: new Date().toISOString(),
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      const avatarDesign = await ollamaService.generateAvatarDescription(
        settings.default_model,
        mockPersona
      );

      setFormData({
        ...formData,
        avatar_config: {
          ...formData.avatar_config,
          primary_color: avatarDesign.primary_color,
          secondary_color: avatarDesign.secondary_color,
          glow_color: avatarDesign.glow_color,
          shape_type: avatarDesign.shape_type as AvatarConfig["shape_type"],
          animation_style: avatarDesign.animation_style as AvatarConfig["animation_style"],
          complexity: avatarDesign.complexity,
        },
      });

      toast.success(`Avatar designed! ${avatarDesign.reasoning}`);
    } catch (error) {
      toast.error("Failed to generate avatar design");
      console.error(error);
    } finally {
      setIsGeneratingAvatar(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div className="space-y-2">
          <Label>Name</Label>
          <Input
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            placeholder="Persona name"
            required
          />
        </div>

        <div className="space-y-2">
          <Label>Description</Label>
          <Input
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            placeholder="Brief description"
          />
        </div>

        <div className="space-y-2">
          <Label>Personality Prompt</Label>
          <Textarea
            value={formData.personality_prompt}
            onChange={(e) => setFormData({ ...formData, personality_prompt: e.target.value })}
            placeholder="Define how this persona behaves and responds..."
            rows={4}
          />
          <p className="text-xs text-muted-foreground">
            This prompt defines the AI's character, behavior, and how it responds to users.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Wake Words</Label>
            <Input
              value={formData.wake_words}
              onChange={(e) => setFormData({ ...formData, wake_words: e.target.value })}
              placeholder="Jarvis, Hey Jarvis, Assistant"
            />
            <p className="text-xs text-muted-foreground">
              Comma-separated phrases to wake the assistant
            </p>
          </div>
          <div className="space-y-2">
            <Label>Response Phrases</Label>
            <Input
              value={formData.response_words}
              onChange={(e) => setFormData({ ...formData, response_words: e.target.value })}
              placeholder="Yes?, I'm here, At your service"
            />
            <p className="text-xs text-muted-foreground">
              Comma-separated responses when woken (random pick)
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Voice Selection (KittenTTS)</Label>
            {hasCreatedVoice && (
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full flex items-center gap-1">
                <Volume2 className="w-3 h-3" />
                Custom voice available
              </span>
            )}
          </div>
          <Select
            value={formData.voice_id}
            onValueChange={(value) => setFormData({ ...formData, voice_id: value })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a KittenTTS voice" />
            </SelectTrigger>
            <SelectContent className="max-h-[300px]">
              {hasCreatedVoice && (
                <SelectItem value="created" className="font-semibold text-primary">
                  🎤 Custom Created Voice (Qwen3-TTS)
                </SelectItem>
              )}
              
              <div className="px-2 py-1 text-xs font-semibold text-muted-foreground bg-muted">
                Female Voices
              </div>
              {VOICE_GROUPS.female.map((voice) => (
                <SelectItem key={voice.value} value={voice.value}>
                  {voice.label}
                </SelectItem>
              ))}
              
              <div className="px-2 py-1 text-xs font-semibold text-muted-foreground bg-muted mt-1">
                Male Voices
              </div>
              {VOICE_GROUPS.male.map((voice) => (
                <SelectItem key={voice.value} value={voice.value}>
                  {voice.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Select a KittenTTS AI voice. For custom voice cloning, use the Voice Studio with Qwen3-TTS.
          </p>
        </div>

        {/* Avatar Type Selection */}
        <div className="space-y-4 border rounded-lg p-4">
          <Label className="flex items-center gap-2 text-sm font-semibold">
            <Sparkles className="w-4 h-4" />
            Avatar Type
          </Label>
          
          <Select 
            value={formData.avatar_config.type || "abstract"}
            onValueChange={(value) => 
              setFormData({
                ...formData,
                avatar_config: { ...formData.avatar_config, type: value as any }
              })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="abstract">
                <div className="flex flex-col items-start">
                  <span>Abstract Sphere</span>
                  <span className="text-xs text-muted-foreground">Animated geometric shape (default)</span>
                </div>
              </SelectItem>
              
              <SelectItem value="vrm">
                <div className="flex flex-col items-start">
                  <span>VRM Avatar</span>
                  <span className="text-xs text-muted-foreground">3D character model (.vrm)</span>
                </div>
              </SelectItem>

            </SelectContent>
          </Select>
          
          
          
          <p className="text-xs text-muted-foreground">
            Choose how your persona appears. VRM avatars support lip sync with speech!
          </p>
          
          {/* VRM Library Selection */}
          {formData.avatar_config.type === "vrm" && (
            <>
              <VrmLibrarySelector
                selectedVrmId={formData.avatar_config.vrm_id}
                onSelect={(vrmId, _modelUrl) => setFormData({
                  ...formData,
                  avatar_config: { 
                    ...formData.avatar_config, 
                    vrm_id: vrmId,
                    // Don't store blob URL - it expires! Just store vrm_id
                    model_url: undefined 
                  }
                })}
              />
              
              {/* VRMA Animation Library */}
              <VrmaLibrarySelector
                personaId={persona?.id}
                vrmaPaths={formData.avatar_config.vrma_paths}
                onChange={(vrmaPaths) => setFormData({
                  ...formData,
                  avatar_config: {
                    ...formData.avatar_config,
                    vrma_paths: vrmaPaths
                  }
                })}
              />
              
              {/* Base Animation Toggles */}
              <BaseAnimationSelector
                enabledVrmas={formData.avatar_config.enabled_base_vrmas}
                onChange={(enabledVrmas) => setFormData({
                  ...formData,
                  avatar_config: {
                    ...formData.avatar_config,
                    enabled_base_vrmas: enabledVrmas
                  }
                })}
              />
            </>
          )}
        </div>

        {/* Color Customization */}
        <div className="space-y-4 border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Palette className="w-4 h-4" />
              <span>Avatar Colors</span>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={generateAvatar}
              disabled={isGeneratingAvatar}
            >
              {isGeneratingAvatar ? (
                <>
                  <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                  Generating...
                </>
              ) : (
                <>
                  <Wand2 className="w-3 h-3 mr-2" />
                  AI Design Avatar
                </>
              )}
            </Button>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label className="text-xs">Primary Color</Label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={formData.avatar_config.primary_color}
                  onChange={(e) => setFormData({
                    ...formData,
                    avatar_config: { ...formData.avatar_config, primary_color: e.target.value }
                  })}
                  className="w-8 h-8 rounded cursor-pointer border-0"
                />
                <span className="text-xs text-muted-foreground font-mono">
                  {formData.avatar_config.primary_color}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Secondary Color</Label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={formData.avatar_config.secondary_color}
                  onChange={(e) => setFormData({
                    ...formData,
                    avatar_config: { ...formData.avatar_config, secondary_color: e.target.value }
                  })}
                  className="w-8 h-8 rounded cursor-pointer border-0"
                />
                <span className="text-xs text-muted-foreground font-mono">
                  {formData.avatar_config.secondary_color}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Glow Color</Label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={formData.avatar_config.glow_color}
                  onChange={(e) => setFormData({
                    ...formData,
                    avatar_config: { ...formData.avatar_config, glow_color: e.target.value }
                  })}
                  className="w-8 h-8 rounded cursor-pointer border-0"
                />
                <span className="text-xs text-muted-foreground font-mono">
                  {formData.avatar_config.glow_color}
                </span>
              </div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground bg-muted p-2 rounded">
            <p>Click "AI Design Avatar" to have the LLM create a unique appearance based on the personality.</p>
            <p>You can manually adjust colors afterward.</p>
          </div>
        </div>
      </div>

      <div className="flex gap-2 justify-end">
        <Button type="button" variant="outline" onClick={onCancel}>
          <X className="w-4 h-4 mr-2" />
          Cancel
        </Button>
        <Button type="submit">
          <Save className="w-4 h-4 mr-2" />
          Save Persona
        </Button>
      </div>
    </form>
  );
}

export function PersonaManager() {
  const { personas, currentPersona, setCurrentPersona, addPersona, updatePersona, deletePersona } = useStore();
  const [editingPersona, setEditingPersona] = useState<Persona | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const handleCreate = (data: Partial<Persona>) => {
    const newPersona: Persona = {
      id: Date.now().toString(),
      name: data.name || "New Persona",
      description: data.description || "",
      personality_prompt: data.personality_prompt || "",
      wake_words: data.wake_words || ["Mimic"],
      response_words: data.response_words || ["Yes?", "I'm listening"],
      voice_id: data.voice_id || "Bella",
      voice_create: null,
      avatar_config: data.avatar_config as AvatarConfig,
      memory: {
        short_term: [],
        long_term: [],
        summary: "",
        last_summarized: new Date().toISOString(),
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    
    addPersona(newPersona);
    setCurrentPersona(newPersona); // Auto-select the new persona
    setIsCreating(false);
    toast.success("Persona created successfully");
  };

  const handleUpdate = (data: Partial<Persona>) => {
    if (!editingPersona) return;
    
    const updated: Persona = {
      ...editingPersona,
      ...data,
      avatar_config: data.avatar_config as AvatarConfig,
      updated_at: new Date().toISOString(),
    };
    
    updatePersona(updated);
    if (currentPersona?.id === updated.id) {
      setCurrentPersona(updated);
    }
    setEditingPersona(null);
    toast.success("Persona updated successfully");
  };

  const handleDelete = (id: string) => {
    if (!confirm("Are you sure you want to delete this persona?")) return;
    deletePersona(id);
    toast.success("Persona deleted successfully");
  };

  const handleActivate = (persona: Persona) => {
    setCurrentPersona(persona);
    toast.success(`Switched to ${persona.name}`);
  };

  return (
    <div className="h-full overflow-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold">Personas</h2>
          <p className="text-muted-foreground">Manage your AI assistant personalities</p>
        </div>
        <Dialog open={isCreating} onOpenChange={setIsCreating}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              New Persona
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-auto">
            <DialogHeader>
              <DialogTitle>Create New Persona</DialogTitle>
            </DialogHeader>
            <PersonaForm
              onSave={handleCreate}
              onCancel={() => setIsCreating(false)}
            />
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid gap-4">
        <AnimatePresence>
          {personas.map((persona) => (
            <motion.div
              key={persona.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className={cn(
                "p-4 rounded-lg border transition-all",
                currentPersona?.id === persona.id
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center"
                    style={{
                      background: `linear-gradient(135deg, ${persona.avatar_config.primary_color}, ${persona.avatar_config.secondary_color})`,
                    }}
                  >
                    <Bot className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold">{persona.name}</h3>
                      {currentPersona?.id === persona.id && (
                        <span className="text-xs bg-primary text-primary-foreground px-2 py-0.5 rounded-full">
                          Active
                        </span>
                      )}
                      {persona.voice_create && (
                        <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full flex items-center gap-1">
                          <Volume2 className="w-3 h-3" />
                          Created Voice
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">{persona.description}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Mic className="w-3 h-3" />
                        Wake: "{persona.wake_words?.[0] || "Mimic"}"
                      </span>
                      <span className="flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        {persona.voice_create ? "Custom Voice" : KITTEN_VOICES.find((v) => v.value === persona.voice_id)?.label || persona.voice_id}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {currentPersona?.id !== persona.id && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleActivate(persona)}
                    >
                      <Check className="w-4 h-4 mr-1" />
                      Activate
                    </Button>
                  )}
                  
                  <Dialog open={editingPersona?.id === persona.id} onOpenChange={(open) => !open && setEditingPersona(null)}>
                    <DialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setEditingPersona(persona)}
                      >
                        <Edit2 className="w-4 h-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl max-h-[90vh] overflow-auto">
                      <DialogHeader>
                        <DialogTitle>Edit Persona</DialogTitle>
                      </DialogHeader>
                      <PersonaForm
                        persona={persona}
                        onSave={handleUpdate}
                        onCancel={() => setEditingPersona(null)}
                      />
                    </DialogContent>
                  </Dialog>

                  {persona.id !== "default" && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDelete(persona.id)}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
