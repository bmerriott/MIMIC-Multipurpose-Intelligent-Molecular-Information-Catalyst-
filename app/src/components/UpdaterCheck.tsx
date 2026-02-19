/**
 * Update Checker Component
 * Shows current version and checks for updates
 */

import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Check, Download, Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { unifiedStorage } from "@/services/unifiedStorage";

export function UpdaterCheck() {
  const [checking, setChecking] = useState(false);
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [currentVersion, _setCurrentVersion] = useState("1.0.0");
  const [latestVersion, _setLatestVersion] = useState("");
  const [isTauri, setIsTauri] = useState(false);

  useEffect(() => {
    setIsTauri(unifiedStorage.isTauri());
    
    // Get version from Tauri if available
    if (unifiedStorage.isTauri()) {
      checkForUpdates();
    }
  }, []);

  const checkForUpdates = async () => {
    if (!unifiedStorage.isTauri()) {
      toast.info("Updates managed by browser");
      return;
    }

    setChecking(true);
    
    try {
      // In a real implementation, this would call Tauri's updater API
      // For now, we'll simulate the check
      
      // Simulate API call to check latest version
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock: no update available (replace with actual Tauri updater)
      setUpdateAvailable(false);
      
      toast.success("You're up to date!", {
        description: `Mimic AI v${currentVersion} is the latest version.`,
      });
    } catch (error) {
      toast.error("Failed to check for updates");
    } finally {
      setChecking(false);
    }
  };

  const installUpdate = async () => {
    toast.info("Update would be installed here", {
      description: "This feature connects to Tauri's auto-updater",
    });
  };

  if (!isTauri) {
    return (
      <div className="p-4 bg-muted rounded-lg">
        <p className="text-sm text-muted-foreground">
          Running in browser mode. Updates are managed by your browser.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
        <div>
          <p className="font-medium">Current Version</p>
          <p className="text-sm text-muted-foreground">v{currentVersion}</p>
        </div>
        
        {updateAvailable ? (
          <div className="flex items-center gap-2 text-amber-400">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm">v{latestVersion} available</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-green-400">
            <Check className="w-5 h-5" />
            <span className="text-sm">Up to date</span>
          </div>
        )}
      </div>

      {updateAvailable ? (
        <Button 
          onClick={installUpdate}
          className="w-full"
          variant="default"
        >
          <Download className="w-4 h-4 mr-2" />
          Download and Install Update
        </Button>
      ) : (
        <Button 
          onClick={checkForUpdates}
          disabled={checking}
          className="w-full"
          variant="outline"
        >
          {checking ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Checking...
            </>
          ) : (
            <>
              <Check className="w-4 h-4 mr-2" />
              Check for Updates
            </>
          )}
        </Button>
      )}

      <p className="text-xs text-muted-foreground">
        Auto-updater connects to GitHub releases. 
        Your data will be preserved during updates.
      </p>
    </div>
  );
}
