import { useState } from "react";
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
  Volume2
} from "lucide-react";
import { useStore } from "@/store";
import type { Persona, AvatarConfig } from "@/types";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Textarea } from "./ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "./ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ollamaService } from "@/services/ollama";

const VOICES = [
  { value: "vivian", label: "Vivian (Chinese Female)" },
  { value: "serena", label: "Serena (Chinese Female)" },
  { value: "uncle_fu", label: "Uncle Fu (Chinese Male)" },
  { value: "dylan", label: "Dylan (Beijing Male)" },
  { value: "eric", label: "Eric (Sichuan Male)" },
  { value: "ryan", label: "Ryan (English Male)" },
  { value: "aiden", label: "Aiden (American Male)" },
  { value: "ono_anna", label: "Ono Anna (Japanese Female)" },
  { value: "sohee", label: "Sohee (Korean Female)" },
];

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
    voice_id: persona?.voice_id || "aiden",
    avatar_config: persona?.avatar_config || {
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
            <Label>Voice Selection</Label>
            {hasCreatedVoice && (
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full flex items-center gap-1">
                <Volume2 className="w-3 h-3" />
                Created voice available
              </span>
            )}
          </div>
          <Select
            value={formData.voice_id}
            onValueChange={(value) => setFormData({ ...formData, voice_id: value })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a voice" />
            </SelectTrigger>
            <SelectContent>
              {hasCreatedVoice && (
                <SelectItem value="created">
                  🎤 Created Voice (Custom)
                </SelectItem>
              )}
              {VOICES.map((voice) => (
                <SelectItem key={voice.value} value={voice.value}>
                  {voice.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Select a preset voice or use the Voice Studio to create a custom voice
          </p>
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
      voice_id: data.voice_id || "aiden",
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
                        {persona.voice_create ? "Custom Created Voice" : VOICES.find((v) => v.value === persona.voice_id)?.label.split(" (")[0]}
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
