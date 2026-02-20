import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "@/store";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Shield, Sparkles, FileText, ExternalLink, Check } from "lucide-react";

import { Sidebar } from "./components/Sidebar";
import { AvatarScene } from "./components/AvatarScene";
import { ChatPanel } from "./components/ChatPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { PersonaManager } from "./components/PersonaManager";
import { VoiceCreator } from "./components/VoiceCreator";
import { Header } from "./components/Header";
import { WakeWordListener } from "./components/WakeWordListener";
import { SpeechDebug } from "./components/SpeechDebug";
import { GlobalAudioPlayer } from "./components/GlobalAudioPlayer";
import { SetupWizard } from "./components/SetupWizard";
import { TOS } from "./legal/TOS";
import { PrivacyPolicy } from "./legal/PrivacyPolicy";
import { ollamaService } from "@/services/ollama";
import { ttsService } from "@/services/tts";
import { unifiedStorage } from "@/services/unifiedStorage";

// First-launch consent state
interface FirstLaunchConsent {
  aiDisclosure: boolean;
  voiceConsent: boolean;
  noMisuse: boolean;
  watermarkAck: boolean;
}

type LegalView = "main" | "tos" | "privacy";

function App() {
  const [activeView, setActiveView] = useState<"chat" | "settings" | "personas" | "voice">("chat");
  const [isLoading, setIsLoading] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [setupComplete, setSetupComplete] = useState(false);
  
  // Legal document viewer
  const [legalView, setLegalView] = useState<LegalView>("main");
  
  // First-launch consent dialog
  const [showFirstLaunchDialog, setShowFirstLaunchDialog] = useState(false);
  const [firstLaunchConsent, setFirstLaunchConsent] = useState<FirstLaunchConsent>({
    aiDisclosure: false,
    voiceConsent: false,
    noMisuse: false,
    watermarkAck: false,
  });
  
  const {
    settings,
    updateAppState,
    updateSettings,
    setAvailableModels,
  } = useStore();

  const allConsentsGiven = firstLaunchConsent.aiDisclosure && 
                           firstLaunchConsent.voiceConsent && 
                           firstLaunchConsent.noMisuse && 
                           firstLaunchConsent.watermarkAck;

  // Apply theme to document
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(settings.theme);
  }, [settings.theme]);

  // Check for first launch
  useEffect(() => {
    const hasAcceptedTerms = localStorage.getItem('mimic_terms_accepted');
    if (!hasAcceptedTerms) {
      setShowFirstLaunchDialog(true);
    }
  }, []);

  // Check if running in Tauri and if setup is needed
  useEffect(() => {
    const checkSetup = async () => {
      // Only show setup wizard in Tauri desktop app
      if (!unifiedStorage.isTauri()) {
        setSetupComplete(true);
        return;
      }
      
      // Check if already completed setup
      const setupDone = localStorage.getItem('mimic_setup_complete');
      if (setupDone) {
        setSetupComplete(true);
        return;
      }
      
      // Check dependencies
      const status = await unifiedStorage.checkSetupStatus();
      if (status?.python_installed && status?.dependencies_installed) {
        localStorage.setItem('mimic_setup_complete', 'true');
        setSetupComplete(true);
      }
      // Otherwise, setup wizard will show
    };
    
    checkSetup();
  }, []);

  const handleAcceptTerms = () => {
    localStorage.setItem('mimic_terms_accepted', 'true');
    localStorage.setItem('mimic_terms_accepted_date', new Date().toISOString());
    setShowFirstLaunchDialog(false);
    toast.success("Welcome to Mimic AI!", {
      description: "You can now start creating AI personas and voices.",
    });
  };

  // Memory usage reminder
  useEffect(() => {
    const memoryTimer = setTimeout(() => {
      toast.info("Memory Usage Reminder", {
        description: "AI services use significant RAM. Close all processes when done to free memory.",
        duration: 10000,
      });
    }, 10 * 60 * 1000); // Show after 10 minutes
    
    return () => clearTimeout(memoryTimer);
  }, []);

  // Initialize connections on startup - NON-BLOCKING
  // Show UI immediately, retry connections in background
  useEffect(() => {
    const checkConnections = async () => {
      // Quick initial check (no retry yet - just get status)
      try {
        ollamaService.setBaseUrl(settings.ollama_url);
        const ollamaConnected = await ollamaService.checkConnection();
        updateAppState({ ollama_connected: ollamaConnected });
        
        if (ollamaConnected) {
          const models = await ollamaService.listModels();
          const modelNames = models.map(m => m.name);
          setAvailableModels(modelNames);
          
          // Auto-select first available model if current default isn't installed
          if (models.length > 0 && !modelNames.includes(settings.default_model)) {
            const firstModel = models[0].name;
            console.log(`[App] Default model ${settings.default_model} not found, switching to ${firstModel}`);
            updateSettings({ default_model: firstModel });
            updateAppState({ current_model: firstModel });
            toast.info(`Model updated`, { 
              description: `Switched to ${firstModel} (llama3.2 not found)` 
            });
          }
          
          // Auto-select vision model if current one isn't available
          const visionModels = modelNames.filter(m => 
            m.includes('llava') || 
            m.includes('bakllava') || 
            m.includes('moondream') || 
            m.includes('vision')
          );
          if (visionModels.length > 0 && !modelNames.includes(settings.vision_model)) {
            const firstVision = visionModels[0];
            console.log(`[App] Vision model ${settings.vision_model} not found, switching to ${firstVision}`);
            updateSettings({ vision_model: firstVision });
          } else if (visionModels.length === 0) {
            // No vision models available - set to empty or default
            updateSettings({ vision_model: 'none' });
          }
        }
      } catch (error) {
        console.log('[App] Ollama not ready (will retry)');
      }

      try {
        ttsService.setBaseUrl(settings.tts_backend_url);
        const ttsConnected = await ttsService.checkConnection();
        updateAppState({ tts_backend_connected: ttsConnected });
      } catch (error) {
        console.log('[App] TTS backend not ready (will retry)');
      }

      // Hide loading screen quickly (after 1 second minimum for UX)
      setTimeout(() => {
        setIsLoading(false);
        
        // Check browser support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          toast.warning("Browser Not Fully Supported", {
            description: "Wake word detection requires Chrome or Edge. You can still use text chat!",
            duration: 8000,
          });
        }
      }, 1000);
    };

    checkConnections();
    
    // Background retry loop - doesn't block UI
    const retryDelay = (attempt: number) => Math.min(2000 * Math.pow(1.3, attempt), 10000);
    
    const backgroundRetry = async () => {
      let attempts = 0;
      const maxAttempts = 15;
      
      while (attempts < maxAttempts) {
        const { appState } = useStore.getState();
        
        // Retry Ollama if not connected
        if (!appState.ollama_connected) {
          try {
            ollamaService.setBaseUrl(settings.ollama_url);
            const connected = await ollamaService.checkConnection();
            if (connected) {
              updateAppState({ ollama_connected: true });
              const models = await ollamaService.listModels();
              const modelNames = models.map(m => m.name);
              setAvailableModels(modelNames);
              
              // Auto-select models on connect
              if (models.length > 0 && !modelNames.includes(settings.default_model)) {
                const firstModel = models[0].name;
                updateSettings({ default_model: firstModel });
                updateAppState({ current_model: firstModel });
                toast.success(`Model auto-selected: ${firstModel}`);
              }
              
              console.log('[App] Ollama connected (background retry)');
            }
          } catch (e) {
            // Silent fail - will retry
          }
        }
        
        // Retry TTS if not connected
        if (!appState.tts_backend_connected) {
          try {
            ttsService.setBaseUrl(settings.tts_backend_url);
            const connected = await ttsService.checkConnection();
            if (connected) {
              updateAppState({ tts_backend_connected: true });
              console.log('[App] TTS backend connected (background retry)');
            }
          } catch (e) {
            // Silent fail - will retry
          }
        }
        
        // Stop if both connected
        const { appState: currentState } = useStore.getState();
        if (currentState.ollama_connected && currentState.tts_backend_connected) {
          break;
        }
        
        attempts++;
        await new Promise(r => setTimeout(r, retryDelay(attempts)));
      }
    };
    
    // Start background retry after initial check
    setTimeout(backgroundRetry, 2000);
    
  }, []);

  // Show setup wizard for Tauri app if not complete
  if (!setupComplete && unifiedStorage.isTauri()) {
    return (
      <SetupWizard onComplete={() => {
        localStorage.setItem('mimic_setup_complete', 'true');
        setSetupComplete(true);
      }} />
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <motion.div 
            className="w-24 h-24 mx-auto mb-6 relative"
            animate={{ 
              scale: [1, 1.05, 1],
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <img 
              src="/mimic.svg" 
              alt="Mimic Logo" 
              className="w-full h-full object-contain drop-shadow-[0_0_15px_rgba(255,0,255,0.6)]"
            />
          </motion.div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#ff00ff] via-[#ff6ac1] to-[#00ffff] bg-clip-text text-transparent drop-shadow-[0_0_10px_rgba(255,0,255,0.5)]">
            Mimic
          </h1>
          <p className="text-[#00ffff] mt-2 text-lg drop-shadow-[0_0_8px_rgba(0,255,255,0.6)]">Initializing AI Assistant...</p>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="text-xs text-[#ff6ac1] mt-4"
          >
            Connecting to Ollama...<br />
            Checking TTS backend...
          </motion.p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <WakeWordListener />
      <GlobalAudioPlayer />
      
      <Sidebar activeView={activeView} onViewChange={setActiveView} />
      
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header 
          onToggleDebug={() => setShowDebug(!showDebug)}
          showDebug={showDebug}
        />
        
        <div className="flex-1 flex min-w-0 overflow-hidden">
          {/* Avatar - takes remaining space, hidden on small screens when panel is shown */}
          {settings.show_avatar && (
            <div className="flex-1 relative min-w-0 hidden md:block">
              <AvatarScene />
            </div>
          )}
          
          {/* Right panel - responsive width */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeView}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className={`
                border-l border-border bg-card/50 backdrop-blur-sm
                flex-shrink-0
                ${settings.show_avatar 
                  ? "w-full md:w-[320px] lg:w-[380px] xl:w-[420px] 2xl:w-[450px]" 
                  : "flex-1"
                }
              `}
            >
              {activeView === "chat" && <ChatPanel />}
              {activeView === "settings" && <SettingsPanel />}
              {activeView === "personas" && <PersonaManager />}
              {activeView === "voice" && <VoiceCreator />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
      
      <Toaster />
      {showDebug && <SpeechDebug onClose={() => setShowDebug(false)} />}

      {/* First Launch Consent Dialog - Modal, cannot be dismissed without accepting */}
      <Dialog 
        open={showFirstLaunchDialog} 
        onOpenChange={(open) => {
          // Prevent closing the dialog unless terms are accepted
          if (!open && !localStorage.getItem('mimic_terms_accepted')) {
            return; // Don't allow closing
          }
          setShowFirstLaunchDialog(open);
        }}
        modal={true}
      >
        <DialogContent 
          className="max-w-lg max-h-[90vh] overflow-hidden border-destructive p-0"
          onPointerDownOutside={(e) => {
            // Prevent closing when clicking outside
            e.preventDefault();
          }}
          onEscapeKeyDown={(e) => {
            // Prevent closing with Escape key
            e.preventDefault();
          }}
          onInteractOutside={(e) => {
            // Prevent any outside interaction from closing
            e.preventDefault();
          }}
          // Hide the close button - must accept terms to proceed
          showCloseButton={false}
        >
          {legalView === "main" ? (
            <div className="flex flex-col max-h-[90vh]">
              {/* Mandatory Notice Banner */}
              <div className="px-6 py-3 bg-destructive/20 border-y border-destructive/50 flex-shrink-0">
                <p className="text-sm text-center font-semibold text-destructive">
                  ⚠️ You must accept these terms to use Mimic AI
                </p>
              </div>
              
              {/* Scrollable Content */}
              <div className="overflow-y-auto flex-1 p-6">
                <DialogHeader className="text-center mb-6">
                  <div className="flex justify-center mb-3">
                    <Sparkles className="w-10 h-10 text-primary" />
                  </div>
                  <DialogTitle className="text-2xl text-center">
                    Welcome to Mimic AI
                  </DialogTitle>
                  <DialogDescription className="text-base text-center">
                    Before you begin, please review and accept the following important information.
                  </DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-4">
                  {/* AI Disclosure */}
                  <div className="p-3 bg-primary/10 rounded-lg space-y-2">
                    <div className="flex items-start gap-2">
                      <Sparkles className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-semibold text-sm">AI-Generated Content</h4>
                        <p className="text-xs text-muted-foreground">
                          Mimic AI uses artificial intelligence to generate responses and synthesize voices. 
                          All AI-generated content should be treated as synthetic.
                        </p>
                      </div>
                    </div>
                    <label className="flex items-start gap-3 cursor-pointer ml-7 p-2 rounded hover:bg-white/5 transition-colors">
                      <Checkbox 
                        checked={firstLaunchConsent.aiDisclosure}
                        onCheckedChange={(checked) => setFirstLaunchConsent(prev => ({ ...prev, aiDisclosure: checked === true }))}
                        className="w-5 h-5 border-2 border-primary data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      />
                      <span className="text-sm">I understand that Mimic AI generates synthetic content</span>
                    </label>
                  </div>

                  {/* Synthetic Voice Notice */}
                  <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg space-y-2">
                    <div className="flex items-start gap-2">
                      <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-semibold text-sm text-green-400">Synthetic Voice Creation</h4>
                        <p className="text-xs text-muted-foreground">
                          Voices are created using AI synthesis with adjustable parameters. All voices are 
                          artificially generated and unique.
                        </p>
                      </div>
                    </div>
                    <label className="flex items-start gap-3 cursor-pointer ml-7 p-2 rounded hover:bg-white/5 transition-colors">
                      <Checkbox 
                        checked={firstLaunchConsent.voiceConsent}
                        onCheckedChange={(checked) => setFirstLaunchConsent(prev => ({ ...prev, voiceConsent: checked === true }))}
                        className="w-5 h-5 border-2 border-green-400 data-[state=checked]:bg-green-400 data-[state=checked]:border-green-400"
                      />
                      <span className="text-sm">I understand that voices are synthetically generated</span>
                    </label>
                  </div>

                  {/* No Misuse */}
                  <div className="p-3 bg-destructive/10 border border-destructive/30 rounded-lg space-y-2">
                    <div className="flex items-start gap-2">
                      <Shield className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-semibold text-sm text-destructive">Prohibited Uses</h4>
                        <p className="text-xs text-muted-foreground">
                          You may NOT use Mimic AI for fraud, impersonation of real individuals, harassment, 
                          creating deceptive content, or any illegal purpose.
                        </p>
                      </div>
                    </div>
                    <label className="flex items-start gap-3 cursor-pointer ml-7 p-2 rounded hover:bg-white/5 transition-colors">
                      <Checkbox 
                        checked={firstLaunchConsent.noMisuse}
                        onCheckedChange={(checked) => setFirstLaunchConsent(prev => ({ ...prev, noMisuse: checked === true }))}
                        className="w-5 h-5 border-2 border-destructive data-[state=checked]:bg-destructive data-[state=checked]:border-destructive"
                      />
                      <span className="text-sm">I agree not to use Mimic AI for fraudulent or harmful purposes</span>
                    </label>
                  </div>

                  {/* Watermark Ack */}
                  <div className="p-3 bg-muted rounded-lg space-y-2">
                    <div className="flex items-start gap-2">
                      <FileText className="w-5 h-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-semibold text-sm">Audio Watermarking</h4>
                        <p className="text-xs text-muted-foreground">
                          All AI-generated audio contains invisible watermarks for identification. 
                          This helps prevent misuse and provides transparency.
                        </p>
                      </div>
                    </div>
                    <label className="flex items-start gap-3 cursor-pointer ml-7 p-2 rounded hover:bg-white/5 transition-colors">
                      <Checkbox 
                        checked={firstLaunchConsent.watermarkAck}
                        onCheckedChange={(checked) => setFirstLaunchConsent(prev => ({ ...prev, watermarkAck: checked === true }))}
                        className="w-5 h-5 border-2 border-muted-foreground data-[state=checked]:bg-muted-foreground data-[state=checked]:border-muted-foreground"
                      />
                      <span className="text-sm">I understand that AI-generated audio is watermarked</span>
                    </label>
                  </div>
                </div>
              </div>

              {/* Footer - Fixed at bottom */}
              <DialogFooter className="flex flex-col items-center justify-center gap-3 p-6 pt-4 border-t border-border bg-card flex-shrink-0 sm:flex-col sm:justify-center">
                {!allConsentsGiven && (
                  <p className="text-xs text-amber-400 text-center w-full">
                    Please check all boxes to continue
                  </p>
                )}
                <Button 
                  onClick={handleAcceptTerms}
                  disabled={!allConsentsGiven}
                  className="w-full"
                  size="lg"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  I Understand & Agree - Start Using Mimic AI
                </Button>
                <div className="text-xs text-muted-foreground text-center w-full flex flex-wrap items-center justify-center gap-1">
                  <span>By clicking above, you agree to the </span>
                  <button 
                    onClick={() => setLegalView("tos")}
                    className="text-primary hover:underline font-medium inline-flex items-center gap-0.5"
                  >
                    Terms of Service
                    <ExternalLink className="w-3 h-3" />
                  </button>
                  <span> and </span>
                  <button 
                    onClick={() => setLegalView("privacy")}
                    className="text-primary hover:underline font-medium inline-flex items-center gap-0.5"
                  >
                    Privacy Policy
                    <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </DialogFooter>
            </div>
          ) : legalView === "tos" ? (
            <div className="h-[80vh] overflow-hidden">
              <TOS onBack={() => setLegalView("main")} />
            </div>
          ) : (
            <div className="h-[80vh] overflow-hidden">
              <PrivacyPolicy onBack={() => setLegalView("main")} />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default App;
