/**
 * Memory Manager Component
 * Provides UI for browsing memory files and confirming write operations
 * Also shows conversation memories from the persistent memory system
 */

import { useState, useEffect, useCallback } from "react";
import {
  BrainCircuit,
  FileText,
  Search,
  Save,
  Trash2,
  X,
  Check,
  AlertCircle,
  RefreshCw,
  Brain,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { memoryToolsService, type MemoryFile, type MemorySearchMatch } from "@/services/memoryTools";
import { useStore } from "@/store";
import { toast } from "sonner";

interface MemoryManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function MemoryManager({ isOpen, onClose }: MemoryManagerProps) {
  const [files, setFiles] = useState<MemoryFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<MemorySearchMatch[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [newFileName, setNewFileName] = useState("");
  const [newFileContent, setNewFileContent] = useState("");
  const [showNewFileDialog, setShowNewFileDialog] = useState(false);
  const [deleteConfirmFile, setDeleteConfirmFile] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState("files");
  const [deleteAllConfirm, setDeleteAllConfirm] = useState<"files" | "conversations" | null>(null);
  const [deleteMemoryConfirm, setDeleteMemoryConfirm] = useState<{personaId: string, index: number, content: string} | null>(null);
  const [selectedPersonaId, setSelectedPersonaId] = useState<string | "all">("current");
  
  // Get personas from store to access their memories
  const { personas, currentPersona, updatePersona } = useStore();

  const loadFiles = useCallback(async () => {
    setIsLoading(true);
    try {
      const fileList = await memoryToolsService.listMemories();
      setFiles(fileList);
    } catch (error: any) {
      toast.error("Failed to load memories", { description: error.message });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      loadFiles();
      // Reset to current persona when opening
      setSelectedPersonaId("current");
    }
  }, [isOpen, loadFiles]);
  
  // Get ALL conversation memories (for tab count)
  const allMemories = personas.flatMap(p => 
    (p.memory?.short_term || []).map((m: any) => ({
      ...m,
      personaName: p.name,
      personaId: p.id,
    }))
  ).sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  
  // Get filtered conversation memories based on selected persona
  const filteredMemories = (() => {
    let targetPersonas: typeof personas;
    if (selectedPersonaId === "all") {
      targetPersonas = personas;
    } else if (selectedPersonaId === "current") {
      targetPersonas = currentPersona ? [currentPersona] : personas;
    } else {
      targetPersonas = personas.filter(p => p.id === selectedPersonaId);
    }
    
    return targetPersonas.flatMap(p => 
      (p.memory?.short_term || []).map((m: any, idx: number) => ({
        ...m,
        personaName: p.name,
        personaId: p.id,
        originalIndex: idx, // Store the actual index in the persona's memory array
      }))
    ).sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  })();
  
  // Get summary for the viewing persona (show for all single persona views)
  const viewingPersonaSummary = (() => {
    if (selectedPersonaId === "all") return "";
    
    const targetPersona = selectedPersonaId === "current" 
      ? currentPersona 
      : personas.find(p => p.id === selectedPersonaId);
    
    return targetPersona?.memory?.summary || "";
  })();
  
  // Get the persona being viewed
  const viewingPersona = selectedPersonaId === "all" 
    ? null 
    : selectedPersonaId === "current" 
      ? currentPersona 
      : personas.find(p => p.id === selectedPersonaId);

  const handleFileSelect = async (filename: string) => {
    setIsLoading(true);
    setSelectedFile(filename);
    try {
      const content = await memoryToolsService.readMemory(filename);
      setFileContent(content);
    } catch (error: any) {
      toast.error("Failed to read file", { description: error.message });
      setFileContent("");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    try {
      const results = await memoryToolsService.searchMemories(searchQuery);
      setSearchResults(results);
    } catch (error: any) {
      toast.error("Search failed", { description: error.message });
    } finally {
      setIsSearching(false);
    }
  };

  const handleSaveNewFile = async () => {
    if (!newFileName.trim()) {
      toast.error("Filename is required");
      return;
    }
    try {
      const result = await memoryToolsService.writeMemory(newFileName, newFileContent, true);
      if (result.success) {
        toast.success("File saved", { description: result.message });
        setShowNewFileDialog(false);
        setNewFileName("");
        setNewFileContent("");
        loadFiles();
      } else {
        toast.error("Failed to save", { description: result.message });
      }
    } catch (error: any) {
      toast.error("Error saving file", { description: error.message });
    }
  };

  const handleDeleteFile = async () => {
    if (!deleteConfirmFile) return;
    try {
      const result = await memoryToolsService.deleteMemory(deleteConfirmFile, true);
      if (result.success) {
        toast.success("File deleted", { description: result.message });
        setDeleteConfirmFile(null);
        if (selectedFile === deleteConfirmFile) {
          setSelectedFile(null);
          setFileContent("");
        }
        loadFiles();
      } else {
        toast.error("Failed to delete", { description: result.message });
      }
    } catch (error: any) {
      toast.error("Error deleting file", { description: error.message });
    }
  };

  const handleDeleteAllFiles = async () => {
    try {
      setIsLoading(true);
      await Promise.all(files.map(file => memoryToolsService.deleteMemory(file.name, true)));
      toast.success(`Deleted ${files.length} memory files`);
      setFiles([]);
      setSelectedFile(null);
      setFileContent("");
      setDeleteAllConfirm(null);
    } catch (error: any) {
      toast.error("Failed to delete all files", { description: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAllConversationMemories = async () => {
    try {
      setIsLoading(true);
      
      // Determine which personas to clear based on filter
      let targetPersonaIds: string[];
      if (selectedPersonaId === "all") {
        targetPersonaIds = personas.map(p => p.id);
      } else if (selectedPersonaId === "current") {
        targetPersonaIds = currentPersona ? [currentPersona.id] : [];
      } else {
        targetPersonaIds = [selectedPersonaId];
      }
      
      // Update only targeted personas
      personas.forEach(p => {
        if (targetPersonaIds.includes(p.id)) {
          updatePersona({
            ...p,
            memory: {
              ...p.memory,
              short_term: [],
              summary: "", // Also clear summary
            },
          });
        }
      });
      
      const personaNames = targetPersonaIds.map(id => personas.find(p => p.id === id)?.name).filter(Boolean);
      toast.success(targetPersonaIds.length === 1 
        ? `Cleared memories for ${personaNames[0]}` 
        : `Cleared memories for ${targetPersonaIds.length} personas`
      );
      setDeleteAllConfirm(null);
    } catch (error: any) {
      toast.error("Failed to delete memories", { description: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteConversationMemory = async (personaId: string, originalIndex: number) => {
    try {
      const persona = personas.find(p => p.id === personaId);
      if (!persona || !persona.memory?.short_term) return;

      const updatedMemory = {
        ...persona.memory,
        short_term: persona.memory.short_term.filter((_, i) => i !== originalIndex),
      };

      updatePersona({
        ...persona,
        memory: updatedMemory,
      });

      toast.success("Memory entry deleted");
      setDeleteMemoryConfirm(null);
    } catch (error: any) {
      toast.error("Failed to delete memory", { description: error.message });
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-6xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BrainCircuit className="w-6 h-6 text-primary" />
            Memory Manager
          </DialogTitle>
          <DialogDescription>
            Browse, search, and manage your AI&apos;s memory files. The AI can read these files to recall information.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="flex-1 flex flex-col min-h-0">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="files" className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Memory Files ({files.length})
            </TabsTrigger>
            <TabsTrigger value="conversations" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Conversation Memories ({allMemories.length})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="files" className="flex-1 flex gap-4 min-h-0 mt-4">
          {/* Left sidebar - File list */}
          <div className="w-1/3 flex flex-col gap-2 border-r pr-4">
            {/* Header with Delete All */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">{files.length} files</span>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={loadFiles} disabled={isLoading}>
                  <RefreshCw className={`w-3 h-3 mr-1 ${isLoading ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
                {files.length > 0 && (
                  <Button 
                    variant="ghost" 
                    size="sm"
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    onClick={() => setDeleteAllConfirm("files")}
                  >
                    <Trash2 className="w-4 h-4 mr-1" />
                    Delete All Files
                  </Button>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Input
                placeholder="Search files..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                className="flex-1"
              />
              <Button variant="outline" size="icon" onClick={handleSearch} disabled={isSearching}>
                <Search className="w-4 h-4" />
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-1 border rounded-lg p-2">
              {searchResults.length > 0 ? (
                // Show search results
                searchResults.map((result) => (
                  <button
                    key={result.filename}
                    onClick={() => handleFileSelect(result.filename)}
                    className={`w-full text-left p-2 rounded text-sm hover:bg-muted transition-colors ${
                      selectedFile === result.filename ? "bg-primary/10 border border-primary/30" : ""
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium truncate">{result.filename}</span>
                    </div>
                    <p className="text-xs text-muted-foreground truncate mt-1">{result.snippet}</p>
                  </button>
                ))
              ) : files.length === 0 ? (
                <p className="text-center text-muted-foreground text-sm py-8">No memory files yet</p>
              ) : (
                // Show all files
                files.map((file) => (
                  <div
                    key={file.name}
                    onClick={() => handleFileSelect(file.name)}
                    className={`w-full text-left p-2 rounded text-sm hover:bg-muted transition-colors group cursor-pointer ${
                      selectedFile === file.name ? "bg-primary/10 border border-primary/30" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 overflow-hidden">
                        <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                        <span className="font-medium truncate">{file.name}</span>
                      </div>
                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteConfirmFile(file.name);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded text-destructive transition-opacity cursor-pointer"
                        role="button"
                      >
                        <Trash2 className="w-3 h-3" />
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground truncate mt-1">{file.preview}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / 1024).toFixed(1)} KB • {new Date(file.modified).toLocaleDateString()}
                    </p>
                  </div>
                ))
              )}
            </div>

            <div className="flex gap-2">
              <Button variant="outline" className="flex-1" onClick={() => setShowNewFileDialog(true)}>
                <Save className="w-4 h-4 mr-2" />
                New Memory File
              </Button>
            </div>
          </div>

          {/* Right side - Content viewer */}
          <div className="flex-1 flex flex-col">
            {selectedFile ? (
              <>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">{selectedFile}</h3>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedFile(null)}>
                    <X className="w-4 h-4" />
                  </Button>
                </div>
                <Textarea
                  value={fileContent}
                  readOnly
                  className="flex-1 font-mono text-sm resize-none"
                  placeholder="File content..."
                />
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
                <FileText className="w-12 h-12 mb-4 opacity-50" />
                <p>Select a file to view its contents</p>
                <p className="text-sm mt-2">The AI can read these files during conversations</p>
              </div>
            )}
          </div>
          </TabsContent>
          
          <TabsContent value="conversations" className="flex-1 flex flex-col min-h-0 mt-4">
            {/* Persona Selector */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-medium">View memories for:</span>
              <select
                value={selectedPersonaId}
                onChange={(e) => setSelectedPersonaId(e.target.value)}
                className="text-sm bg-muted border rounded px-2 py-1 flex-1"
              >
                <option value="current">Current Persona ({currentPersona?.name || "None"})</option>
                {personas.map(p => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
                <option value="all">All Personas</option>
              </select>
            </div>
            
            {/* Summary Section - Show for single persona view */}
            {viewingPersonaSummary && selectedPersonaId !== "all" && (
              <div className="mb-4 p-3 bg-muted rounded-lg">
                <h4 className="text-sm font-semibold mb-1">Conversation Summary</h4>
                <p className="text-sm text-muted-foreground">{viewingPersonaSummary}</p>
              </div>
            )}
            
            {/* Header with Delete All */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">
                {filteredMemories.length} memories
                {selectedPersonaId === "all" && ` across ${personas.length} personas`}
                {selectedPersonaId !== "all" && viewingPersona && ` for ${viewingPersona.name}`}
              </span>
              {filteredMemories.length > 0 && (
                <Button 
                  variant="ghost" 
                  size="sm"
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                  onClick={() => setDeleteAllConfirm("conversations")}
                >
                  <Trash2 className="w-4 h-4 mr-1" />
                  Delete All Memories
                </Button>
              )}
            </div>
            
            {/* Memory List */}
            <div className="flex-1 overflow-y-auto border rounded-lg p-2">
              {filteredMemories.length === 0 ? (
                <p className="text-center text-muted-foreground text-sm py-8">
                  No conversation memories found for {selectedPersonaId === "all" ? "any persona" : "this persona"}.
                  <br />Enable memory in settings to start building context.
                </p>
              ) : (
                <div className="space-y-2">
                  {filteredMemories.map((memory: any) => (
                    <div key={`${memory.personaId}-${memory.originalIndex}`} className="p-3 bg-muted rounded-lg text-sm group">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-xs text-primary">{memory.personaName}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">
                            {new Date(memory.timestamp).toLocaleString()}
                          </span>
                          <button
                            onClick={() => setDeleteMemoryConfirm({
                              personaId: memory.personaId,
                              index: memory.originalIndex,
                              content: memory.content.slice(0, 50) + (memory.content.length > 50 ? "..." : ""),
                            })}
                            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded text-destructive transition-opacity"
                            title="Delete this memory"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                      <p className="text-muted-foreground line-clamp-3">{memory.content}</p>
                      {memory.importance > 0.8 && (
                        <span className="inline-flex items-center gap-1 mt-1 text-xs text-amber-500">
                          <AlertCircle className="w-3 h-3" />
                          High importance
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        {/* New File Dialog */}
        <Dialog open={showNewFileDialog} onOpenChange={setShowNewFileDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Memory File</DialogTitle>
              <DialogDescription>Save information for the AI to reference later</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Filename</label>
                <Input
                  placeholder="e.g., project_notes.txt"
                  value={newFileName}
                  onChange={(e) => setNewFileName(e.target.value)}
                />
              </div>
              <div>
                <label className="text-sm font-medium">Content</label>
                <Textarea
                  placeholder="Enter the content..."
                  value={newFileContent}
                  onChange={(e) => setNewFileContent(e.target.value)}
                  rows={8}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowNewFileDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleSaveNewFile}>
                <Save className="w-4 h-4 mr-2" />
                Save Memory
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Confirmation Dialog */}
        <Dialog open={!!deleteConfirmFile} onOpenChange={() => setDeleteConfirmFile(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                Delete Memory File?
              </DialogTitle>
              <DialogDescription>
                Are you sure you want to delete &quot;{deleteConfirmFile}&quot;? This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteConfirmFile(null)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteFile}>
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete All Files Confirmation */}
        <Dialog open={deleteAllConfirm === "files"} onOpenChange={() => setDeleteAllConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                Delete All Memory Files?
              </DialogTitle>
              <DialogDescription>
                Are you sure you want to delete all {files.length} memory files? This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteAllConfirm(null)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteAllFiles}>
                <Trash2 className="w-4 h-4 mr-2" />
                Delete All
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete All Conversation Memories Confirmation */}
        <Dialog open={deleteAllConfirm === "conversations"} onOpenChange={() => setDeleteAllConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                Clear {selectedPersonaId === "all" ? "All" : "Selected"} Conversation Memories?
              </DialogTitle>
              <DialogDescription>
                Are you sure you want to delete {filteredMemories.length} conversation memories
                {selectedPersonaId === "all" 
                  ? " across all personas" 
                  : selectedPersonaId === "current" 
                    ? ` for ${currentPersona?.name || "current persona"}`
                    : ` for ${viewingPersona?.name || "selected persona"}`
                }? This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteAllConfirm(null)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteAllConversationMemories}>
                <Trash2 className="w-4 h-4 mr-2" />
                Clear {selectedPersonaId === "all" ? "All" : "Selected"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Individual Memory Confirmation */}
        <Dialog open={!!deleteMemoryConfirm} onOpenChange={() => setDeleteMemoryConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                Delete Memory Entry?
              </DialogTitle>
              <DialogDescription>
                Delete this memory from {deleteMemoryConfirm ? personas.find(p => p.id === deleteMemoryConfirm.personaId)?.name : ""}?
                <br /><br />
                <span className="text-xs text-muted-foreground italic">&quot;{deleteMemoryConfirm?.content}&quot;</span>
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteMemoryConfirm(null)}>
                Cancel
              </Button>
              <Button 
                variant="destructive" 
                onClick={() => deleteMemoryConfirm && handleDeleteConversationMemory(deleteMemoryConfirm.personaId, deleteMemoryConfirm.index)}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Pending Confirmation Dialog
 * Shows when the AI wants to write to a file
 */
interface ConfirmationDialogProps {
  isOpen: boolean;
  filename: string;
  content?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function MemoryWriteConfirmation({
  isOpen,
  filename,
  content,
  onConfirm,
  onCancel,
}: ConfirmationDialogProps) {
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-amber-500" />
            Confirm Memory Write
          </DialogTitle>
          <DialogDescription>
            The AI wants to save information to your memory folder. Review before confirming.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium">Filename</label>
            <div className="p-2 bg-muted rounded text-sm font-mono">{filename}</div>
          </div>

          {content && (
            <div>
              <label className="text-sm font-medium">Content Preview</label>
              <Textarea
                value={content.slice(0, 500) + (content.length > 500 ? "..." : "")}
                readOnly
                rows={6}
                className="font-mono text-sm"
              />
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            <X className="w-4 h-4 mr-2" />
            Cancel
          </Button>
          <Button onClick={onConfirm}>
            <Check className="w-4 h-4 mr-2" />
            Confirm Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
