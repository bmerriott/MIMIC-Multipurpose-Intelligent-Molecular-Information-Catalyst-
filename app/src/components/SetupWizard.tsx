/**
 * Setup Wizard Component
 * Handles first-time setup: checking dependencies and auto-install
 */

import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/shell";
import { listen } from "@tauri-apps/api/event";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { AlertCircle, Check, Download, Loader2, ExternalLink } from "lucide-react";

interface DependencyStatus {
  python_installed: boolean;
  python_version?: string;
  python_path?: string;
  pip_installed: boolean;
  dependencies_installed: boolean;
  missing_packages: string[];
  venv_exists: boolean;
}

interface InstallProgress {
  stage: string;
  message: string;
  percent: number;
  is_complete: boolean;
  error?: string;
}

interface SetupWizardProps {
  onComplete: () => void;
}

export function SetupWizard({ onComplete }: SetupWizardProps) {
  const [status, setStatus] = useState<DependencyStatus | null>(null);
  const [progress, setProgress] = useState<InstallProgress | null>(null);
  const [isInstalling, setIsInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkDependencies();
  }, []);

  useEffect(() => {
    // Listen for install progress events
    const unlisten = listen<InstallProgress>("install-progress", (event) => {
      setProgress(event.payload);
    });

    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  const checkDependencies = async () => {
    try {
      const deps = await invoke<DependencyStatus>("check_dependencies");
      setStatus(deps);
      
      // If everything is ready, complete immediately
      if (deps.python_installed && deps.dependencies_installed) {
        setTimeout(onComplete, 500);
      }
    } catch (e) {
      setError("Failed to check dependencies");
    }
  };

  const startInstall = async () => {
    setIsInstalling(true);
    setError(null);
    
    try {
      await invoke("install_dependencies_command");
      // Installation complete
      onComplete();
    } catch (e) {
      setError(e as string);
      setIsInstalling(false);
    }
  };

  const openPythonDownload = () => {
    open("https://www.python.org/downloads/");
  };

  if (!status) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Checking system...</p>
        </div>
      </div>
    );
  }

  // Python not installed
  if (!status.python_installed) {
    return (
      <div className="flex items-center justify-center h-screen bg-background p-4">
        <div className="max-w-md w-full bg-card border rounded-lg p-6 shadow-lg">
          <div className="flex items-center gap-3 mb-4">
            <AlertCircle className="w-8 h-8 text-amber-500" />
            <h1 className="text-xl font-bold">Python Required</h1>
          </div>
          
          <p className="text-muted-foreground mb-6">
            Mimic AI requires Python 3.10 or higher to run the voice synthesis engine. 
            Python was not detected on your system.
          </p>

          <div className="space-y-3">
            <Button 
              onClick={openPythonDownload}
              className="w-full"
              variant="default"
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              Download Python
            </Button>
            
            <Button 
              onClick={checkDependencies}
              className="w-full"
              variant="outline"
            >
              Check Again
            </Button>
          </div>

          <p className="text-xs text-muted-foreground mt-4">
            After installing Python, click "Check Again". Make sure to check "Add Python to PATH" during installation.
          </p>
        </div>
      </div>
    );
  }

  // Installing dependencies
  if (isInstalling || (progress && !progress.is_complete)) {
    return (
      <div className="flex items-center justify-center h-screen bg-background p-4">
        <div className="max-w-md w-full bg-card border rounded-lg p-6 shadow-lg">
          <h1 className="text-xl font-bold mb-2">Setting Up Mimic AI</h1>
          <p className="text-muted-foreground text-sm mb-6">
            Installing voice synthesis dependencies. This may take 5-10 minutes.
          </p>

          <div className="space-y-4">
            <Progress value={progress?.percent || 0} className="h-2" />
            
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              <p className="text-sm">{progress?.message || "Preparing installation..."}</p>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Ready to install dependencies
  if (!status.dependencies_installed && !isInstalling) {
    return (
      <div className="flex items-center justify-center h-screen bg-background p-4">
        <div className="max-w-md w-full bg-card border rounded-lg p-6 shadow-lg">
          <div className="flex items-center gap-3 mb-4">
            <Download className="w-8 h-8 text-primary" />
            <h1 className="text-xl font-bold">Install Dependencies</h1>
          </div>
          
          <p className="text-muted-foreground mb-4">
            Found {status.python_version}. Now we need to install the voice synthesis packages.
          </p>

          <div className="bg-muted rounded-lg p-3 mb-4">
            <p className="text-xs text-muted-foreground mb-2">Packages to install:</p>
            <div className="flex flex-wrap gap-1">
              {status.missing_packages.slice(0, 5).map((pkg) => (
                <span key={pkg} className="text-xs bg-background px-2 py-0.5 rounded">
                  {pkg}
                </span>
              ))}
              {status.missing_packages.length > 5 && (
                <span className="text-xs text-muted-foreground">
                  +{status.missing_packages.length - 5} more
                </span>
              )}
            </div>
          </div>

          <Button 
            onClick={startInstall}
            className="w-full"
            size="lg"
          >
            <Download className="w-4 h-4 mr-2" />
            Install Dependencies
          </Button>

          <p className="text-xs text-muted-foreground mt-4">
            This will create a virtual environment and install packages. Internet connection required.
          </p>
        </div>
      </div>
    );
  }

  // All done
  return (
    <div className="flex items-center justify-center h-screen bg-background p-4">
      <div className="max-w-md w-full bg-card border rounded-lg p-6 shadow-lg text-center">
        <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
          <Check className="w-8 h-8 text-green-500" />
        </div>
        <h1 className="text-xl font-bold mb-2">Ready!</h1>
        <p className="text-muted-foreground mb-4">
          All dependencies are installed. Starting Mimic AI...
        </p>
        <Loader2 className="w-5 h-5 animate-spin mx-auto text-primary" />
      </div>
    </div>
  );
}
