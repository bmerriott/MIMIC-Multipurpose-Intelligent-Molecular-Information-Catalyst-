/**
 * Subscription Dialog - Trust Model
 * 
 * Friendly reminder about supporting development
 * - No enforcement, no license keys
 * - User can dismiss and continue using
 * - Trial countdown during trial period
 */

import { useState, useEffect } from 'react';
import { 
  checkSubscription, 
  getPatreonUrl as _getPatreonUrl, 
  markAsSupporter,
  isDevMode,
  type SubscriptionState 
} from '@/services/subscriptionService';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ExternalLink, Heart, Clock, Sparkles } from 'lucide-react';

interface SubscriptionDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onContinue?: () => void;
}

export function SubscriptionDialog({ open, onOpenChange, onContinue }: SubscriptionDialogProps) {
  const [subState, setSubState] = useState<SubscriptionState | null>(null);
  const [showThanks, setShowThanks] = useState(false);

  useEffect(() => {
    if (open) {
      loadSubscriptionState();
    }
  }, [open]);

  const loadSubscriptionState = async () => {
    const state = await checkSubscription();
    setSubState(state);
    
    // Auto-close if in dev mode
    if (isDevMode()) {
      onContinue?.();
      onOpenChange(false);
    }
  };

  const handlePatreonClick = () => {
    markAsSupporter();
    window.open('https://www.patreon.com/c/MimicAIDigitalAssistant', '_blank');
    setShowThanks(true);
    setTimeout(() => {
      onContinue?.();
      onOpenChange(false);
    }, 2000);
  };

  const handleContinueAnyway = () => {
    onContinue?.();
    onOpenChange(false);
  };

  const handleMaybeLater = () => {
    onContinue?.();
    onOpenChange(false);
  };

  // Don't show if dev mode (handled in load)
  if (isDevMode()) return null;

  const isTrialActive = subState?.status === 'trial_active';
  const isExpired = subState?.status === 'trial_expired';

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
              {isTrialActive ? (
                <Clock className="w-6 h-6 text-white" />
              ) : (
                <Heart className="w-6 h-6 text-white" />
              )}
            </div>
            <div>
              <DialogTitle>
                {isTrialActive ? 'Free Trial' : 'Support Mimic AI'}
              </DialogTitle>
              <DialogDescription>
                {subState?.message}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-4 pt-4">
          {/* Trial Progress */}
          {isTrialActive && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Trial Progress</span>
                <span className="font-medium">{subState.daysRemaining} days left</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all"
                  style={{ 
                    width: `${(subState.daysRemaining / subState.trialTotalDays) * 100}%` 
                  }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Enjoying Mimic AI? Consider supporting development on Patreon!
              </p>
            </div>
          )}

          {/* Support Section */}
          {(isTrialActive || isExpired) && (
            <div className="space-y-3">
              <Button 
                onClick={handlePatreonClick}
                className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Support on Patreon ($5/month)
              </Button>
              
              <p className="text-xs text-center text-muted-foreground">
                Your support helps fund ongoing development!
              </p>
            </div>
          )}

          {/* Thanks Message */}
          {showThanks && (
            <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-green-500" />
              <span className="text-sm text-green-700">Thank you for your support!</span>
            </div>
          )}

          {/* Continue Options */}
          <div className="pt-2 border-t space-y-2">
            {isExpired ? (
              <>
                <Button 
                  variant="outline" 
                  onClick={handleContinueAnyway}
                  className="w-full"
                >
                  Continue Using (Trust Model)
                </Button>
                <p className="text-xs text-center text-muted-foreground">
                  Mimic AI uses an honor system. You can continue using the app,
                  but please consider supporting if you find it valuable.
                </p>
              </>
            ) : (
              <Button 
                variant="outline" 
                onClick={handleMaybeLater}
                className="w-full"
              >
                {isTrialActive ? 'Continue Using' : 'Maybe Later'}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Hook to check subscription on app start
export function useSubscriptionCheck() {
  const [showDialog, setShowDialog] = useState(false);
  const [subState, setSubState] = useState<SubscriptionState | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    const check = async () => {
      // Skip if dev mode
      if (isDevMode()) {
        setChecked(true);
        return;
      }

      const state = await checkSubscription();
      setSubState(state);
      
      // Show dialog if:
      // - Trial just expired (show nag)
      // - First launch (trial started, show info)
      if (state.showNag || state.status === 'trial_active') {
        setShowDialog(true);
      }
      
      setChecked(true);
    };

    check();
  }, []);

  return {
    showDialog,
    setShowDialog,
    subState,
    checked,
    isDevMode: isDevMode(),
  };
}
