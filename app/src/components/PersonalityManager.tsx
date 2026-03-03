/**
 * Personality Manager Component
 * 
 * Allows users to view and manage what their persona has learned about:
 * - Its own personality traits
 * - Skills it has developed
 * - User preferences it has observed
 * - Relationship history
 * 
 * This gives users transparency and control over the AI's evolving personality.
 */

import { useState, useMemo } from "react";
import { 
  Brain, 
  Check, 
  Trash2, 
  Sparkles, 
  TrendingUp, 
  User, 
  Heart,
  Lightbulb,
  AlertCircle
} from "lucide-react";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";
import { useStore } from "@/store";
import { personalityLearning, type LearnedInsight } from "@/services/personalityLearning";

// Safe content conversion helper
const safeContent = (content: any): string => {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (typeof content === "object") {
    try {
      return JSON.stringify(content);
    } catch {
      return String(content);
    }
  }
  return String(content);
};
import { toast } from "sonner";

interface PersonalityManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

const TYPE_ICONS = {
  trait: Sparkles,
  skill: TrendingUp,
  preference: User,
  relationship: Heart,
  fact: Lightbulb,
};

const TYPE_COLORS = {
  trait: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  skill: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  preference: "bg-green-500/20 text-green-400 border-green-500/30",
  relationship: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  fact: "bg-amber-500/20 text-amber-400 border-amber-500/30",
};

export function PersonalityManager({ isOpen, onClose }: PersonalityManagerProps) {
  const { currentPersona, updatePersona } = useStore();
  const [selectedTab, setSelectedTab] = useState<"all" | "trait" | "skill" | "preference" | "relationship">("all");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const learningData = currentPersona?.learning_data;
  const allInsights: LearnedInsight[] = learningData?.user_preferences?.insights || [];

  // Filter active (not removed) insights
  const activeInsights = useMemo(() => {
    return allInsights.filter((i) => !i.userRemoved);
  }, [allInsights]);

  // Filter by selected tab
  const filteredInsights = useMemo(() => {
    if (selectedTab === "all") return activeInsights;
    return activeInsights.filter((i) => i.type === selectedTab);
  }, [activeInsights, selectedTab]);

  // Group by type for counts
  const counts = useMemo(() => ({
    all: activeInsights.length,
    trait: activeInsights.filter((i) => i.type === "trait").length,
    skill: activeInsights.filter((i) => i.type === "skill").length,
    preference: activeInsights.filter((i) => i.type === "preference").length,
    relationship: activeInsights.filter((i) => i.type === "relationship").length,
  }), [activeInsights]);

  // Get relationship depth
  const relationshipDepth = learningData 
    ? personalityLearning.getRelationshipDepth(learningData)
    : 0;

  const handleVerify = (insight: LearnedInsight) => {
    if (!currentPersona || !learningData) return;

    const updatedData = personalityLearning.verifyInsight(learningData, insight.id);
    updatePersona({
      ...currentPersona,
      learning_data: updatedData,
    });

    toast.success("Insight verified", {
      description: "This learning will be reinforced in future conversations.",
    });
  };

  const handleDelete = (insight: LearnedInsight) => {
    if (!currentPersona || !learningData) return;

    const updatedData = personalityLearning.removeInsight(learningData, insight.id);
    updatePersona({
      ...currentPersona,
      learning_data: updatedData,
    });

    setDeleteConfirmId(null);
    toast.success("Insight removed", {
      description: "This learning will no longer influence the persona.",
    });
  };

  if (!currentPersona) {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-primary" />
            {currentPersona.name}&apos;s Personality Development
          </DialogTitle>
          <DialogDescription>
            View and manage what {currentPersona.name} has learned from your conversations.
            Verified insights are used to personalize responses.
          </DialogDescription>
        </DialogHeader>

        {/* Relationship Stats */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="p-3 bg-muted rounded-lg">
            <div className="text-sm text-muted-foreground">Conversations</div>
            <div className="text-2xl font-bold">
              {learningData?.milestones?.conversations_count || 0}
            </div>
          </div>
          <div className="p-3 bg-muted rounded-lg">
            <div className="text-sm text-muted-foreground">Time Together</div>
            <div className="text-2xl font-bold">
              {Math.round((learningData?.total_conversation_time || 0) / 60)}h
            </div>
          </div>
          <div className="p-3 bg-muted rounded-lg">
            <div className="text-sm text-muted-foreground">Relationship Depth</div>
            <div className="flex items-center gap-2">
              <div className="text-2xl font-bold">{Math.round(relationshipDepth * 100)}%</div>
              <div className="flex-1 h-2 bg-muted-foreground/20 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-500"
                  style={{ width: `${relationshipDepth * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Insights Tabs */}
        <Tabs value={selectedTab} onValueChange={(v) => setSelectedTab(v as any)} className="flex-1 flex flex-col min-h-0">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="all">
              All ({counts.all})
            </TabsTrigger>
            <TabsTrigger value="trait">
              Traits ({counts.trait})
            </TabsTrigger>
            <TabsTrigger value="skill">
              Skills ({counts.skill})
            </TabsTrigger>
            <TabsTrigger value="preference">
              Preferences ({counts.preference})
            </TabsTrigger>
            <TabsTrigger value="relationship">
              History ({counts.relationship})
            </TabsTrigger>
          </TabsList>

          <TabsContent value={selectedTab} className="flex-1 overflow-y-auto mt-4 min-h-0">
            {filteredInsights.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground py-8">
                <Brain className="w-12 h-12 mb-4 opacity-50" />
                <p className="text-center">
                  {selectedTab === "all" 
                    ? `No learned insights yet. Have more conversations with ${currentPersona.name} to build their personality.`
                    : `No ${selectedTab} insights learned yet.`
                  }
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {filteredInsights.map((insight) => {
                  const Icon = TYPE_ICONS[insight.type];
                  return (
                    <div 
                      key={insight.id}
                      className={`p-4 rounded-lg border transition-all ${
                        insight.userVerified 
                          ? "bg-primary/5 border-primary/30" 
                          : "bg-muted/50 border-muted"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg ${TYPE_COLORS[insight.type]}`}>
                          <Icon className="w-4 h-4" />
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline" className={TYPE_COLORS[insight.type]}>
                              {insight.type}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {insight.category}
                            </span>
                            {insight.userVerified && (
                              <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/30">
                                <Check className="w-3 h-3 mr-1" />
                                Verified
                              </Badge>
                            )}
                          </div>
                          
                          <p className="text-sm">{safeContent(insight.content)}</p>
                          
                          <div className="flex items-center justify-between mt-2">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <span>Confidence: {Math.round(insight.confidence * 100)}%</span>
                              <span>•</span>
                              <span>{new Date(insight.timestamp).toLocaleDateString()}</span>
                            </div>
                            
                            <div className="flex items-center gap-1">
                              {!insight.userVerified && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-7 text-green-400 hover:text-green-300 hover:bg-green-500/10"
                                  onClick={() => handleVerify(insight)}
                                >
                                  <Check className="w-3 h-3 mr-1" />
                                  Verify
                                </Button>
                              )}
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                                onClick={() => setDeleteConfirmId(insight.id)}
                              >
                                <Trash2 className="w-3 h-3 mr-1" />
                                Remove
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Info Footer */}
        <div className="mt-4 p-3 bg-muted/50 rounded-lg flex items-start gap-2 text-sm text-muted-foreground">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium text-foreground">How this works</p>
            <p>
              {currentPersona.name} learns from your conversations to become more personalized. 
              Verified insights are actively used. Unverified insights may still influence behavior subtly. 
              Removed insights are permanently deleted.
            </p>
          </div>
        </div>

        {/* Delete Confirmation Dialog */}
        <Dialog open={!!deleteConfirmId} onOpenChange={() => setDeleteConfirmId(null)}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                Remove Learning?
              </DialogTitle>
              <DialogDescription>
                This insight will be permanently removed and will no longer influence {currentPersona.name}&apos;s personality.
              </DialogDescription>
            </DialogHeader>
            <div className="flex justify-end gap-2 mt-4">
              <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>
                Cancel
              </Button>
              <Button 
                variant="destructive"
                onClick={() => {
                  const insight = allInsights.find((i) => i.id === deleteConfirmId);
                  if (insight) handleDelete(insight);
                }}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Remove
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}
