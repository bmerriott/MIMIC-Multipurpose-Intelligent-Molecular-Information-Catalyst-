import { useState, useEffect } from "react";
import { 
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Smile, Sparkles, Play } from "lucide-react";
import { toast } from "sonner";

interface EmoteMenuProps {
  // Base VRMAs available globally
  baseVrmas: Record<string, string>;
  // Persona-specific VRMAs
  personaVrmas?: Record<string, string>;
  // Currently enabled base VRMAs
  enabledBaseVrmas: string[];
  // Auto animation enabled
  autoAnimation: boolean;
  // Callbacks
  onToggleBaseVrma: (name: string, enabled: boolean) => void;
  onToggleAutoAnimation: (enabled: boolean) => void;
  onPlayEmote: (name: string) => void;
}

export function EmoteMenu({
  baseVrmas,
  personaVrmas = {},
  enabledBaseVrmas,
  autoAnimation,
  onToggleBaseVrma,
  onToggleAutoAnimation,
  onPlayEmote,
}: EmoteMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [availableEmotes, setAvailableEmotes] = useState<string[]>([]);

  // Build list of available emotes
  useEffect(() => {
    const emotes: string[] = [];
    
    // Add enabled base emotes
    enabledBaseVrmas.forEach(name => {
      if (name !== 'idle' && baseVrmas[name]) {
        emotes.push(name);
      }
    });
    
    // Add persona-specific emotes
    Object.keys(personaVrmas).forEach(name => {
      if (name !== 'idle' && !emotes.includes(name)) {
        emotes.push(name);
      }
    });
    
    setAvailableEmotes(emotes);
  }, [baseVrmas, personaVrmas, enabledBaseVrmas]);

  const handlePlayEmote = (name: string) => {
    onPlayEmote(name);
    toast.success(`Playing ${name}`);
  };

  // Format emote name for display
  const formatEmoteName = (name: string) => {
    return name
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="secondary"
          size="sm"
          className="absolute top-4 left-4 z-10 bg-background/80 backdrop-blur-sm hover:bg-background/90"
        >
          <Smile className="w-4 h-4 mr-1" />
          Emotes
        </Button>
      </PopoverTrigger>
      <PopoverContent 
        className="w-72 p-4" 
        align="start"
        side="right"
        sideOffset={10}
      >
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-sm flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              Animation Controls
            </h4>
          </div>

          {/* Auto Animation Toggle */}
          <div className="flex items-center justify-between py-2 border-b">
            <div className="space-y-0.5">
              <Label className="text-sm">Auto Animation</Label>
              <p className="text-xs text-muted-foreground">
                Play random emotes every 30s
              </p>
            </div>
            <Switch
              checked={autoAnimation}
              onCheckedChange={onToggleAutoAnimation}
            />
          </div>

          {/* On-Demand Emotes */}
          {availableEmotes.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs font-semibold text-muted-foreground">
                Play Emote
              </Label>
              <div className="grid grid-cols-2 gap-1.5">
                {availableEmotes.map((emote) => (
                  <Button
                    key={emote}
                    variant="ghost"
                    size="sm"
                    className="justify-start text-xs h-8 px-2"
                    onClick={() => handlePlayEmote(emote)}
                  >
                    <Play className="w-3 h-3 mr-1 text-primary" />
                    {formatEmoteName(emote)}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Base VRMA Toggles */}
          <div className="space-y-2 pt-2 border-t">
            <Label className="text-xs font-semibold text-muted-foreground">
              Available Base Animations
            </Label>
            <div className="space-y-1.5 max-h-32 overflow-y-auto">
              {Object.keys(baseVrmas)
                .filter(name => name !== 'idle')
                .map((name) => (
                  <div
                    key={name}
                    className="flex items-center justify-between py-0.5"
                  >
                    <span className="text-xs">{formatEmoteName(name)}</span>
                    <Switch
                      checked={enabledBaseVrmas.includes(name)}
                      onCheckedChange={(checked) => onToggleBaseVrma(name, checked)}
                      className="scale-75"
                    />
                  </div>
                ))}
            </div>
          </div>

          {/* Persona-specific emotes info */}
          {Object.keys(personaVrmas).length > 0 && (
            <div className="pt-2 border-t">
              <p className="text-xs text-muted-foreground">
                <Sparkles className="w-3 h-3 inline mr-1" />
                {Object.keys(personaVrmas).length} custom animation(s) available
              </p>
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
