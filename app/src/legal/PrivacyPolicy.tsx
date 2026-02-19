import { motion } from "framer-motion";
import { ArrowLeft, Shield, Server, Database, Lock, ExternalLink, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface PrivacyPolicyProps {
  onBack: () => void;
}

export function PrivacyPolicy({ onBack }: PrivacyPolicyProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center gap-4 p-4 border-b border-border">
        <Button variant="ghost" size="icon" onClick={onBack}>
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            Privacy Policy
          </h1>
          <p className="text-xs text-muted-foreground">Last Updated: February 2026</p>
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="max-w-3xl mx-auto space-y-8 p-6 pb-8">
          
          {/* Explicit Local Processing Disclosure */}
          <section className="space-y-3">
            <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-green-400 mb-3">
                <Server className="w-5 h-5" />
                Local Processing Disclosure
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                <strong className="text-foreground">
                  This application runs entirely locally on your device. We do not collect, store, 
                  or process your voice data, conversations, or generated content.
                </strong> 
                All AI processing, voice synthesis, and data storage occurs on your local machine. 
                We have no ability to access, monitor, or retrieve any data you create or process 
                using this Software.
              </p>
            </div>
          </section>

          {/* Introduction */}
          <section className="space-y-3">
            <p className="text-sm text-muted-foreground leading-relaxed">
              Mimic AI is committed to protecting your privacy. This Privacy Policy explains how we 
              handle your data. <strong>Importantly, Mimic AI does not operate any servers or cloud 
              infrastructure.</strong> The software runs entirely on your local machine.
            </p>
          </section>

          {/* No Servers Notice */}
          <section className="space-y-3">
            <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-green-400">
                <Server className="w-5 h-5" />
                No Central Servers - You Control Your Data
              </h2>
              <p className="text-sm text-muted-foreground mt-2">
                Mimic AI is a desktop application that runs entirely on your computer. We do not 
                operate any servers that store or process your data. All processing happens locally 
                on your device.
              </p>
            </div>
          </section>

          {/* Information We Collect */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Database className="w-4 h-4 text-primary" />
              Information We Collect
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong>Mimic AI does not collect any personal information.</strong> All data is stored 
              locally on your device. This includes:
            </p>
            <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-4">
              <li><strong>Synthetic Voice Configurations:</strong> Voice parameter settings (pitch, speed, etc.) are stored locally</li>
              <li><strong>Conversation Data:</strong> Chat messages are stored locally on your device</li>
              <li><strong>Persona Data:</strong> Character configurations are stored locally</li>
              <li><strong>Settings:</strong> Application preferences are stored locally</li>
            </ul>
            <p className="text-sm text-muted-foreground leading-relaxed mt-3">
              <strong>Important:</strong> We do not record, store, or process voice recordings from you 
              or any other person. Voices are created synthetically using AI parameters only.
            </p>
          </section>

          {/* How We Use Information */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">How We Use Information</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Since all data is local, we do not use your data for any purpose. The Software uses your data solely for:
            </p>
            <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-4">
              <li>Voice parameters: Generating synthetic speech on your device</li>
              <li>Conversation data: Maintaining your chat history locally</li>
              <li>Settings: Remembering your preferences</li>
            </ul>
          </section>

          {/* Data Sharing */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Data Sharing</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong>We do not share, sell, or transmit your data to any third parties.</strong> 
               All data remains on your local device. We have no ability to access your data.
            </p>
          </section>

          {/* Third-Party Services */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <ExternalLink className="w-4 h-4 text-amber-400" />
              Third-Party Dependencies
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              While Mimic AI itself does not collect data, the Software relies on several third-party 
              components and services. By using Mimic AI, you are also subject to the privacy policies 
              of these dependencies:
            </p>

            {/* Ollama */}
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <h3 className="font-semibold text-sm">Ollama (AI Language Models)</h3>
              <p className="text-sm text-muted-foreground">
                Mimic AI connects to Ollama for AI language model functionality. Ollama typically 
                runs locally, but if you use remote Ollama instances or models, your prompts and 
                responses may be processed according to 
                <a 
                  href="https://ollama.com/privacy" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary hover:underline ml-1"
                >
                  Ollama's Privacy Policy
                </a>.
              </p>
              <p className="text-xs text-amber-400">
                <AlertTriangle className="w-3 h-3 inline mr-1" />
                If you share images or files with models that are not locally installed, you are 
                subject to the privacy policy of Ollama and its online services.
              </p>
            </div>

            {/* StyleTTS 2 */}
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <h3 className="font-semibold text-sm">StyleTTS 2/QWEN3 TTS (Voice Synthesis)</h3>
              <p className="text-sm text-muted-foreground">
                Voice synthesis is performed using StyleTTS 2/QWEN3 TTS, which runs entirely locally on your 
                machine. No voice data or parameters are sent to external servers for synthesis.
              </p>
            </div>

            {/* PyTorch */}
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <h3 className="font-semibold text-sm">PyTorch (Machine Learning Framework)</h3>
              <p className="text-sm text-muted-foreground">
                AI models run using PyTorch. By default, PyTorch does not transmit data, but you 
                should review 
                <a 
                  href="https://pytorch.org/legal/privacy-policy" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary hover:underline ml-1"
                >
                  their privacy policy
                </a> 
                if you have concerns.
              </p>
            </div>
          </section>

          {/* Privacy Risks */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold flex items-center gap-2 text-amber-400">
              <AlertTriangle className="w-4 h-4" />
              Potential Privacy Considerations
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              While Mimic AI prioritizes privacy through local processing, users should be aware of 
              the following:
            </p>
            
            <div className="space-y-3">
              <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                <h3 className="font-semibold text-sm text-amber-400">Local AI Models</h3>
                <p className="text-sm text-muted-foreground">
                  AI models (downloaded via Ollama) run locally on your machine. However, these models 
                  may have been trained on public data. Do not input sensitive personal information 
                  expecting complete privacy, as model behavior can be unpredictable. (If cloud models are used data is being sent to Ollama Servers)
                </p>
              </div>

              <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                <h3 className="font-semibold text-sm text-amber-400">Data Storage</h3>
                <p className="text-sm text-muted-foreground">
                  All data is stored locally on your device. This means:
                </p>
                <ul className="list-disc list-inside text-sm text-muted-foreground mt-1 ml-4">
                  <li>You are responsible for securing your local device</li>
                  <li>Other users of your computer may access the data</li>
                  <li>Backups may contain this data if you back up your user folder</li>
                  <li>Data persists until you manually delete it or uninstall the application</li>
                </ul>
              </div>

              <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                <h3 className="font-semibold text-sm text-amber-400">Network Dependencies</h3>
                <p className="text-sm text-muted-foreground">
                  While voice synthesis and chat work locally, downloading models, updates, or using 
                  optional services requires internet access and may involve data transmission 
                  to those respective services.
                </p>
              </div>
            </div>
          </section>

          {/* Data Security */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Lock className="w-4 h-4 text-primary" />
              Data Security
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              We implement the following security measures:
            </p>
            <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-4">
              <li>All voice synthesis happens locally - no audio data leaves your device</li>
              <li>No voice recordings are stored or processed</li>
              <li>All AI-generated audio contains identifying watermarks</li>
              <li>No cloud storage of sensitive data</li>
              <li>Open-source code for transparency</li>
            </ul>
            <p className="text-sm text-muted-foreground leading-relaxed mt-3">
              However, no security is perfect. You are responsible for:
            </p>
            <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-4">
              <li>Securing your local device with strong passwords</li>
              <li>Encrypting your storage if required</li>
              <li>Controlling physical access to your computer</li>
              <li>Managing backups appropriately</li>
            </ul>
          </section>

          {/* Your Rights */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Your Rights</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Because all data is local, you have complete control:
            </p>
            <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-4">
              <li><strong>Access:</strong> All your data is accessible in the application</li>
              <li><strong>Deletion:</strong> Delete any data through the application interface</li>
              <li><strong>Portability:</strong> Export functionality available within the app</li>
              <li><strong>Opt-out:</strong> Uninstall the application to remove all data</li>
            </ul>
          </section>

          {/* Children's Privacy */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Children's Privacy</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Mimic AI is not intended for users under 18 years of age. We do not knowingly collect 
              data from children. Parents or guardians who believe their child has used the Software 
              should contact us to request deletion of any data.
            </p>
          </section>

          {/* Changes to Privacy Policy */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Changes to This Privacy Policy</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              We may update this Privacy Policy from time to time. Changes will be posted here with 
              an updated date. We encourage you to review this policy periodically.
            </p>
          </section>

          {/* Contact */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Contact Us</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              For privacy-related questions, concerns, or to report issues, please contact us through 
              the project's official GitHub repository or community channels.
            </p>
          </section>

          {/* Summary */}
          <div className="pt-8 border-t border-border">
            <div className="p-4 bg-primary/10 rounded-lg">
              <h3 className="font-semibold text-primary mb-2">Summary</h3>
              <p className="text-sm text-muted-foreground">
                Mimic AI is designed with privacy as a core principle. We don't want your data—it's 
                yours. Everything runs locally on your machine. We don't store recorded voices.
                The Software is simply a tool we provide; you maintain full control and 
                responsibility for your data and how you use the Software.
              </p>
            </div>
          </div>

        </div>
      </ScrollArea>
    </motion.div>
  );
}

export default PrivacyPolicy;
