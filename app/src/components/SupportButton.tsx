/**
 * Support Button - Wiggling Patreon link
 * Replaces the automatic subscription nag with an opt-in button
 */

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Heart } from 'lucide-react';
import { motion } from 'framer-motion';

export function SupportButton() {
  const [isWiggling, setIsWiggling] = useState(false);

  // Trigger wiggle every 30-60 seconds randomly
  useEffect(() => {
    const triggerWiggle = () => {
      setIsWiggling(true);
      setTimeout(() => setIsWiggling(false), 1000);
    };

    // Initial wiggle after 10 seconds
    const initialTimeout = setTimeout(triggerWiggle, 10000);

    // Random wiggles
    const scheduleNextWiggle = () => {
      const delay = 30000 + Math.random() * 30000; // 30-60 seconds
      return setTimeout(() => {
        triggerWiggle();
        scheduleNextWiggle();
      }, delay);
    };

    const interval = scheduleNextWiggle();

    return () => {
      clearTimeout(initialTimeout);
      clearTimeout(interval);
    };
  }, []);

  const handleClick = () => {
    window.open('https://www.patreon.com/c/MimicAIDigitalAssistant', '_blank');
  };

  return (
    <motion.div
      animate={isWiggling ? {
        rotate: [0, -5, 5, -5, 5, 0],
        scale: [1, 1.1, 1],
      } : {}}
      transition={{ duration: 0.5 }}
    >
      <Button
        onClick={handleClick}
        variant="ghost"
        size="sm"
        className="gap-2 text-amber-500 hover:text-amber-600 hover:bg-amber-500/10"
      >
        <Heart className="w-4 h-4 fill-current" />
        Support
      </Button>
    </motion.div>
  );
}
