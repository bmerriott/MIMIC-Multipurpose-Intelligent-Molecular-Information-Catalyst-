/**
 * Tool Confirmation Modal
 * 
 * Shows a preview of tool operations requiring user confirmation.
 * Displays exact command and changes before execution.
 */

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle, Check, X, Loader2 } from "lucide-react";
import type { PendingToolConfirmation } from "@/hooks/useToolConfirmation";

interface ToolConfirmationModalProps {
  pending: PendingToolConfirmation | null;
  isExecuting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ToolConfirmationModal({
  pending,
  isExecuting,
  onConfirm,
  onCancel,
}: ToolConfirmationModalProps) {
  if (!pending) return null;

  return (
    <Dialog open={!!pending} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-amber-500">
            <AlertCircle className="w-5 h-5" />
            {pending.preview.title}
          </DialogTitle>
          <DialogDescription>{pending.preview.description}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-muted-foreground mb-1 block">
              Operation Details:
            </label>
            <div className="bg-muted rounded-lg p-3">
              <div className="text-xs text-muted-foreground mb-1">
                Tool: <span className="font-mono text-foreground">{pending.toolName}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Requested by: <span className="text-foreground">{pending.personaName}</span>
              </div>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-muted-foreground mb-1 block">
              Preview of Changes:
            </label>
            <Textarea
              value={pending.preview.content}
              readOnly
              className="font-mono text-xs bg-muted/50 resize-none"
              rows={8}
            />
          </div>

          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
            <p className="text-xs text-amber-400">
              <strong>Security Notice:</strong> This operation will modify files or data. 
              Review the preview carefully before confirming. You can cancel if unsure.
            </p>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button
            variant="outline"
            onClick={onCancel}
            disabled={isExecuting}
            className="flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            Cancel
          </Button>
          <Button
            onClick={onConfirm}
            disabled={isExecuting}
            className="flex items-center gap-2 bg-amber-500 hover:bg-amber-600"
          >
            {isExecuting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <Check className="w-4 h-4" />
                Confirm & Execute
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
