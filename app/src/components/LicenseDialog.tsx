/**
 * License Dialog Component for Mimic AI
 * 
 * Displays:
 * - Current license/trial status
 * - Days remaining
 * - License activation form
 * - Link to Patreon subscription
 */

import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Alert, AlertDescription } from './ui/alert';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { licenseManager } from '../services/licenseManager';
import type { LicenseVerificationResult } from '../services/licenseTypes';
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle, 
  ExternalLink,
  Key,
  Gift
} from 'lucide-react';

interface LicenseDialogProps {
  isOpen: boolean;
  onClose: () => void;
  forceShow?: boolean; // Show even if licensed (for settings)
}

export const LicenseDialog: React.FC<LicenseDialogProps> = ({
  isOpen,
  onClose,
  forceShow = false,
}) => {
  const [licenseStatus, setLicenseStatus] = useState<LicenseVerificationResult | null>(null);
  const [licenseKey, setLicenseKey] = useState('');
  const [isActivating, setIsActivating] = useState(false);
  const [activationMessage, setActivationMessage] = useState('');
  const [activationSuccess, setActivationSuccess] = useState(false);
  const [machineId, setMachineId] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadLicenseStatus();
    }
  }, [isOpen]);

  const loadLicenseStatus = async () => {
    const status = await licenseManager.checkLicenseStatus();
    setLicenseStatus(status);
    
    // Get machine ID for support purposes
    const mid = await licenseManager.getCurrentMachineId();
    setMachineId(mid);
  };

  const handleActivate = async () => {
    if (!licenseKey.trim()) return;

    setIsActivating(true);
    setActivationMessage('');
    setActivationSuccess(false);

    try {
      const result = await licenseManager.activateLicense(licenseKey.trim());
      setActivationMessage(result.message);
      setActivationSuccess(result.success);

      if (result.success) {
        // Reload status after successful activation
        await loadLicenseStatus();
        setLicenseKey('');
      }
    } catch (error) {
      setActivationMessage('An error occurred during activation. Please try again.');
      setActivationSuccess(false);
    } finally {
      setIsActivating(false);
    }
  };

  const handleDeactivate = async () => {
    await licenseManager.deactivateLicense();
    await loadLicenseStatus();
  };

  const openPatreon = () => {
    window.open(licenseManager.getPatreonUrl(), '_blank');
  };

  const openGitHub = () => {
    window.open(licenseManager.getGitHubReleasesUrl(), '_blank');
  };

  const getStatusIcon = () => {
    if (!licenseStatus) return null;
    
    switch (licenseStatus.status) {
      case 'trial_active':
        return <Gift className="w-8 h-8 text-green-500" />;
      case 'trial_expired':
        return <Clock className="w-8 h-8 text-red-500" />;
      case 'licensed_active':
        return <CheckCircle className="w-8 h-8 text-green-500" />;
      case 'licensed_expired':
        return <AlertTriangle className="w-8 h-8 text-orange-500" />;
      default:
        return <XCircle className="w-8 h-8 text-red-500" />;
    }
  };

  const getStatusTitle = () => {
    if (!licenseStatus) return 'Loading...';
    
    switch (licenseStatus.status) {
      case 'trial_active':
        return 'Free Trial Active';
      case 'trial_expired':
        return 'Trial Expired';
      case 'licensed_active':
        return 'Licensed';
      case 'licensed_expired':
        return 'License Expired';
      default:
        return 'Invalid License';
    }
  };

  const getStatusColor = () => {
    if (!licenseStatus) return 'text-gray-500';
    
    switch (licenseStatus.status) {
      case 'trial_active':
      case 'licensed_active':
        return 'text-green-600';
      case 'trial_expired':
      case 'licensed_expired':
        return 'text-red-600';
      default:
        return 'text-red-600';
    }
  };

  // Don't show if licensed and not forced
  if (licenseStatus?.status === 'licensed_active' && !forceShow) {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <DialogTitle className={`text-xl ${getStatusColor()}`}>
                {getStatusTitle()}
              </DialogTitle>
              <DialogDescription>
                {licenseStatus?.message || 'Checking license status...'}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        {/* Trial Status */}
        {licenseStatus?.status === 'trial_active' && (
          <Card className="bg-green-50 border-green-200">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-green-800">
                <Gift className="w-5 h-5" />
                <span className="font-medium">
                  {licenseStatus.daysRemaining} day{licenseStatus.daysRemaining !== 1 ? 's' : ''} remaining in your free trial
                </span>
              </div>
              <p className="mt-2 text-sm text-green-700">
                Enjoy full access to all features! After your trial, subscribe on Patreon to continue using Mimic AI.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Trial Expired */}
        {licenseStatus?.status === 'trial_expired' && (
          <Alert variant="destructive">
            <AlertTriangle className="w-4 h-4" />
            <AlertDescription>
              Your 7-day free trial has ended. Subscribe on Patreon to continue using Mimic AI.
            </AlertDescription>
          </Alert>
        )}

        {/* License Active */}
        {licenseStatus?.status === 'licensed_active' && (
          <Card className="bg-green-50 border-green-200">
            <CardHeader>
              <CardTitle className="text-green-800 flex items-center gap-2">
                <CheckCircle className="w-5 h-5" />
                License Active
              </CardTitle>
              <CardDescription className="text-green-700">
                {licenseStatus.daysRemaining} day{licenseStatus.daysRemaining !== 1 ? 's' : ''} remaining until renewal
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                variant="outline" 
                onClick={handleDeactivate}
                className="w-full"
              >
                Deactivate License
              </Button>
            </CardContent>
          </Card>
        )}

        {/* License Activation Form */}
        {(licenseStatus?.status === 'trial_expired' || 
          licenseStatus?.status === 'licensed_expired' || 
          forceShow) && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="w-5 h-5" />
                Activate License
              </CardTitle>
              <CardDescription>
                Enter your license key from Patreon to activate Mimic AI
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="MIMIC-XXXX-XXXX-XXXX-XXXX"
                  value={licenseKey}
                  onChange={(e) => setLicenseKey(e.target.value.toUpperCase())}
                  className="font-mono uppercase"
                  maxLength={24}
                />
                <Button 
                  onClick={handleActivate} 
                  disabled={isActivating || !licenseKey.trim()}
                >
                  {isActivating ? 'Activating...' : 'Activate'}
                </Button>
              </div>

              {activationMessage && (
                <Alert variant={activationSuccess ? 'default' : 'destructive'}>
                  {activationSuccess ? (
                    <CheckCircle className="w-4 h-4" />
                  ) : (
                    <XCircle className="w-4 h-4" />
                  )}
                  <AlertDescription>{activationMessage}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        )}

        {/* Patreon CTA */}
        {(licenseStatus?.status === 'trial_expired' || 
          licenseStatus?.status === 'licensed_expired') && (
          <Card className="bg-gradient-to-r from-orange-50 to-red-50 border-orange-200">
            <CardHeader>
              <CardTitle className="text-orange-800">Get a License Key</CardTitle>
              <CardDescription className="text-orange-700">
                Support Mimic AI development and get unlimited access
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-orange-900">Patreon Subscription</p>
                  <p className="text-sm text-orange-700">$5/month - Cancel anytime</p>
                </div>
                <Button 
                  onClick={openPatreon}
                  className="bg-orange-600 hover:bg-orange-700 text-white"
                >
                  Subscribe <ExternalLink className="w-4 h-4 ml-2" />
                </Button>
              </div>
              <p className="text-xs text-orange-600">
                After subscribing, you'll receive a license key via email or Patreon messages.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Support Info */}
        <div className="text-xs text-gray-500 pt-4 border-t">
          <p>Having trouble? Contact support with your Machine ID:</p>
          <code className="block mt-1 p-2 bg-gray-100 rounded font-mono text-xs break-all">
            {machineId || 'Loading...'}
          </code>
        </div>

        {/* Footer */}
        <div className="flex justify-between pt-4">
          <Button variant="ghost" size="sm" onClick={openGitHub}>
            GitHub <ExternalLink className="w-3 h-3 ml-1" />
          </Button>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default LicenseDialog;
