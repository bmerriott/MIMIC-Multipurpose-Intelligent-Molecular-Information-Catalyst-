/**
 * Memory Manager Component
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
  History,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "./ui/dialog";
import { Badge } from "./ui/badge";
import { memoryToolsService, type MemoryFile, type MemorySearchMatch } from "@/services/memoryTools";
import { useStore } from "@/store";
import { toast } from "sonner";

interface MemoryManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

// Safe date parsing
const safeParseDate = (timestamp: string | undefined): Date => {
  if (!timestamp) return new Date(0);
  try {
    const date = new Date(timestamp);
    return isNaN(date.getTime()) ? new Date(0) : date;
  } catch {
    return new Date(0);
  }
};

// Safe content conversion - handles objects, arrays, etc.
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

// Display memory type
interface DisplayMemory {
  id: string;
  content: string;
  timestamp: string;
  importance?: number;
  personaName: string;
  personaId: string;
  originalIndex: number;
}

// Tab types
type TabType = "files" | "conversational" | "history";

export function MemoryManager({ isOpen, onClose }: MemoryManagerProps) {
  const [activeTab, setActiveTab] = useState<TabType>("files");
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
  const [deleteAllConfirm, setDeleteAllConfirm] = useState<"files" | "conversations" | null>(null);
  const [deleteMemoryConfirm, setDeleteMemoryConfirm] = useState<{personaId: string, index: number, content: string} | null>(null);
  const [selectedPersonaId, setSelectedPersonaId] = useState<string | "all" | "current">("current");
  
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
      setSelectedPersonaId("current");
      setActiveTab("files");
    }
  }, [isOpen, loadFiles]);
  
  // Get counts safely
  const allMemoriesCount = personas.reduce((count, p) => count + (p.memory?.short_term?.length || 0), 0);
  const currentPersonaMemoriesCount = currentPersona?.memory?.short_term?.length || 0;
  
  // Get filtered memories for history tab
  const getFilteredMemories = (): DisplayMemory[] => {
    try {
      let targetPersonas = personas;
      if (selectedPersonaId === "current" && currentPersona) {
        targetPersonas = [currentPersona];
      } else if (selectedPersonaId !== "all" && selectedPersonaId !== "current") {
        targetPersonas = personas.filter(p => p.id === selectedPersonaId);
      }
      
      const memories: DisplayMemory[] = [];
      targetPersonas.forEach(p => {
        const shortTerm = p.memory?.short_term;
        if (Array.isArray(shortTerm)) {
          shortTerm.forEach((m, idx) => {
            // Skip invalid entries
            if (!m || typeof m !== 'object') return;
            if (!m.id && !m.content && !m.timestamp) return; // Must have at least one valid field
            
            memories.push({
              id: m.id || `${p.id}-${idx}-${Date.now()}`,
              content: safeContent(m.content),
              timestamp: String(m.timestamp || new Date().toISOString()),
              importance: Number(m.importance) || 0.5,
              personaName: String(p.name || "Unknown"),
              personaId: String(p.id),
              originalIndex: idx,
            });
          });
        }
      });
      
      return memories.sort((a, b) => 
        safeParseDate(b.timestamp).getTime() - safeParseDate(a.timestamp).getTime()
      );
    } catch (e) {
      console.error("Error filtering memories:", e);
      return [];
    }
  };
  
  // Get current persona memories
  const getCurrentPersonaMemories = (): DisplayMemory[] => {
    if (!currentPersona) return [];
    const shortTerm = currentPersona.memory?.short_term;
    if (!Array.isArray(shortTerm)) return [];
    
    return shortTerm
      .filter((m) => m && typeof m === 'object' && (m.id || m.content || m.timestamp))
      .map((m, idx) => ({
        id: m.id || `${currentPersona.id}-${idx}-${Date.now()}`,
        content: safeContent(m.content),
        timestamp: String(m.timestamp || new Date().toISOString()),
        importance: Number(m.importance) || 0.5,
        personaName: String(currentPersona.name || "Unknown"),
        personaId: String(currentPersona.id),
        originalIndex: idx,
      })).sort((a, b) => 
      safeParseDate(b.timestamp).getTime() - safeParseDate(a.timestamp).getTime()
    );
  };
  
  const filteredMemories = getFilteredMemories();
  const currentPersonaMemories = getCurrentPersonaMemories();
  
  // Get viewing persona summary
  const viewingPersonaSummary = (() => {
    if (selectedPersonaId === "all") return "";
    const targetPersona = selectedPersonaId === "current" 
      ? currentPersona 
      : personas.find(p => p.id === selectedPersonaId);
    return safeContent(targetPersona?.memory?.summary);
  })();

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
      const result = await memoryToolsService.writeMemory(newFileName, newFileContent, "default", true);
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
      const result = await memoryToolsService.deleteMemory(deleteConfirmFile, "default", true);
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
      await Promise.all(files.map(file => memoryToolsService.deleteMemory(file.name, "default", true)));
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
      let targetPersonaIds: string[];
      if (selectedPersonaId === "all") {
        targetPersonaIds = personas.map(p => p.id);
      } else if (selectedPersonaId === "current") {
        targetPersonaIds = currentPersona ? [currentPersona.id] : [];
      } else {
        targetPersonaIds = [selectedPersonaId];
      }
      
      personas.forEach(p => {
        if (targetPersonaIds.includes(p.id)) {
          updatePersona({
            ...p,
            memory: { ...p.memory, short_term: [], summary: "" },
          });
        }
      });
      
      toast.success("Memories cleared");
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

      updatePersona({
        ...persona,
        memory: {
          ...persona.memory,
          short_term: persona.memory.short_term.filter((_, i) => i !== originalIndex),
        },
      });

      toast.success("Memory deleted");
      setDeleteMemoryConfirm(null);
    } catch (error: any) {
      toast.error("Failed to delete memory", { description: error.message });
    }
  };

  // Render tab button
  const TabButton = ({ tab, icon: Icon, label, count }: { tab: TabType; icon: any; label: string; count: number }) => (
    <button
      onClick={() => setActiveTab(tab)}
      className={`flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors rounded-lg ${
        activeTab === tab 
          ? "bg-primary text-primary-foreground" 
          : "text-muted-foreground hover:bg-muted hover:text-foreground"
      }`}
    >
      <Icon className="w-4 h-4" />
      <span className="hidden sm:inline">{label}</span>
      <span className="sm:hidden">{label.slice(0, 4)}</span>
      <Badge variant={activeTab === tab ? "secondary" : "outline"} className="ml-1 text-xs">
        {count}
      </Badge>
    </button>
  );

  if (!currentPersona) return null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col p-0">
        {/* Header */}
        <div className="p-6 pb-4 border-b flex-shrink-0">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-xl">
              <BrainCircuit className="w-6 h-6 text-primary" />
              Memory Manager
            </DialogTitle>
            <DialogDescription className="text-sm text-muted-foreground">
              Browse and manage memories. Saved files can be read by the AI. Conversational memories are private.
            </DialogDescription>
          </DialogHeader>
        </div>

        {/* Tab Navigation */}
        <div className="px-6 pt-4 border-b flex-shrink-0">
          <div className="grid grid-cols-3 gap-2">
            <TabButton 
              tab="files" 
              icon={FileText} 
              label="Saved Files" 
              count={files.length} 
            />
            <TabButton 
              tab="conversational" 
              icon={Brain} 
              label="Conversational" 
              count={currentPersonaMemoriesCount} 
            />
            <TabButton 
              tab="history" 
              icon={History} 
              label="Full History" 
              count={allMemoriesCount} 
            />
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden p-6">
          
          {/* FILES TAB */}
          {activeTab === "files" && (
            <div className="h-full flex gap-4">
              {/* File List Sidebar */}
              <div className="w-72 flex flex-col gap-3 border-r pr-4 flex-shrink-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">{files.length} files</span>
                  <div className="flex items-center gap-1">
                    <Button variant="ghost" size="icon" onClick={loadFiles} disabled={isLoading} className="h-8 w-8">
                      <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
                    </Button>
                    {files.length > 0 && (
                      <Button 
                        variant="ghost" 
                        size="icon"
                        className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => setDeleteAllConfirm("files")}
                      >
                        <Trash2 className="w-4 h-4" />
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
                    className="flex-1 h-9"
                  />
                  <Button variant="outline" size="icon" onClick={handleSearch} disabled={isSearching} className="h-9 w-9 flex-shrink-0">
                    <Search className="w-4 h-4" />
                  </Button>
                </div>

                <div className="flex-1 overflow-y-auto border rounded-lg p-2 min-h-0">
                  {searchResults.length > 0 ? (
                    searchResults.map((result) => (
                      <button
                        key={result.filename}
                        onClick={() => handleFileSelect(result.filename)}
                        className={`w-full text-left p-2 rounded text-sm hover:bg-muted transition-colors mb-1 ${
                          selectedFile === result.filename ? "bg-primary/10 border border-primary/30" : ""
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                          <span className="font-medium truncate">{result.filename}</span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate mt-1">{result.snippet}</p>
                      </button>
                    ))
                  ) : files.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-center p-4">
                      <FileText className="w-8 h-8 text-muted-foreground mb-2 opacity-50" />
                      <p className="text-sm text-muted-foreground">No files yet</p>
                    </div>
                  ) : (
                    files.map((file) => (
                      <div
                        key={file.name}
                        onClick={() => handleFileSelect(file.name)}
                        className={`w-full text-left p-2 rounded text-sm hover:bg-muted transition-colors group cursor-pointer mb-1 ${
                          selectedFile === file.name ? "bg-primary/10 border border-primary/30" : ""
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 overflow-hidden">
                            <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                            <span className="font-medium truncate">{file.name}</span>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); setDeleteConfirmFile(file.name); }}
                            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded text-destructive transition-opacity flex-shrink-0"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                        <p className="text-xs text-muted-foreground truncate">{file.preview}</p>
                      </div>
                    ))
                  )}
                </div>

                <Button variant="outline" className="w-full" onClick={() => setShowNewFileDialog(true)}>
                  <Save className="w-4 h-4 mr-2" />
                  New File
                </Button>
              </div>

              {/* File Content */}
              <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                {selectedFile ? (
                  <>
                    <div className="flex items-center justify-between mb-3 flex-shrink-0">
                      <h3 className="font-semibold truncate">{selectedFile}</h3>
                      <Button variant="ghost" size="sm" onClick={() => setSelectedFile(null)}>
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                    <Textarea
                      value={fileContent}
                      readOnly
                      className="flex-1 font-mono text-sm resize-none min-h-0"
                    />
                  </>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
                    <FileText className="w-12 h-12 mb-4 opacity-50" />
                    <p>Select a file to view contents</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* CONVERSATIONAL TAB */}
          {activeTab === "conversational" && (
            <div className="h-full flex flex-col">
              <div className="mb-4 p-3 bg-primary/10 border border-primary/20 rounded-lg flex-shrink-0">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary flex-shrink-0" />
                  <span className="text-sm">
                    <span className="font-medium text-primary">Private:</span>{" "}
                    <span className="text-muted-foreground">
                      Only {currentPersona?.name || "this persona"} can access these
                    </span>
                  </span>
                </div>
              </div>
              
              {currentPersona?.memory?.summary && (
                <div className="mb-4 p-3 bg-muted rounded-lg flex-shrink-0">
                  <h4 className="text-sm font-semibold mb-1">Summary</h4>
                  <p className="text-sm text-muted-foreground">{safeContent(currentPersona.memory.summary)}</p>
                </div>
              )}
              
              <div className="flex items-center justify-between mb-3 flex-shrink-0">
                <span className="text-sm text-muted-foreground">{currentPersonaMemories.length} memories</span>
                {currentPersonaMemories.length > 0 && (
                  <Button 
                    variant="ghost" 
                    size="sm"
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    onClick={() => setDeleteAllConfirm("conversations")}
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear All
                  </Button>
                )}
              </div>
              
              <div className="flex-1 overflow-y-auto border rounded-lg p-3 min-h-0">
                {currentPersonaMemories.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <Brain className="w-10 h-10 text-muted-foreground mb-3 opacity-50" />
                    <p className="text-muted-foreground">No memories yet</p>
                    <p className="text-sm text-muted-foreground">Have conversations to build memory</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {currentPersonaMemories.map((memory) => (
                      <div key={memory.id} className="p-3 bg-muted rounded-lg text-sm group">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-xs text-primary">{memory.personaName}</span>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground">
                              {safeParseDate(memory.timestamp).toLocaleString()}
                            </span>
                            <button
                              onClick={() => setDeleteMemoryConfirm({
                                personaId: memory.personaId,
                                index: memory.originalIndex,
                                content: safeContent(memory.content).slice(0, 50) + (safeContent(memory.content).length > 50 ? "..." : ""),
                              })}
                              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded text-destructive transition-opacity"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                        <p className="text-muted-foreground">{safeContent(memory.content)}</p>
                        {(memory.importance || 0) > 0.8 && (
                          <span className="inline-flex items-center gap-1 mt-2 text-xs text-amber-500">
                            <AlertCircle className="w-3 h-3" />
                            High importance
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* HISTORY TAB */}
          {activeTab === "history" && (
            <div className="h-full flex flex-col">
              <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg flex-shrink-0">
                <div className="flex items-center gap-2">
                  <History className="w-4 h-4 text-amber-500 flex-shrink-0" />
                  <span className="text-sm">
                    <span className="font-medium text-amber-500">Admin View:</span>{" "}
                    <span className="text-muted-foreground">Personas cannot see each other&apos;s memories</span>
                  </span>
                </div>
              </div>
              
              <div className="flex items-center gap-3 mb-4 flex-shrink-0">
                <span className="text-sm font-medium whitespace-nowrap">View:</span>
                <select
                  value={selectedPersonaId}
                  onChange={(e) => setSelectedPersonaId(e.target.value)}
                  className="text-sm bg-muted border rounded px-3 py-2 flex-1"
                >
                  <option value="current">Current ({currentPersona?.name || "None"})</option>
                  {personas.map(p => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                  <option value="all">All Personas</option>
                </select>
              </div>
              
              {viewingPersonaSummary && selectedPersonaId !== "all" && (
                <div className="mb-4 p-3 bg-muted rounded-lg flex-shrink-0">
                  <h4 className="text-sm font-semibold mb-1">Summary</h4>
                  <p className="text-sm text-muted-foreground">{viewingPersonaSummary}</p>
                </div>
              )}
              
              <div className="flex items-center justify-between mb-3 flex-shrink-0">
                <span className="text-sm text-muted-foreground">{filteredMemories.length} memories</span>
                {filteredMemories.length > 0 && (
                  <Button 
                    variant="ghost" 
                    size="sm"
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    onClick={() => setDeleteAllConfirm("conversations")}
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear
                  </Button>
                )}
              </div>
              
              <div className="flex-1 overflow-y-auto border rounded-lg p-3 min-h-0">
                {filteredMemories.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <History className="w-10 h-10 text-muted-foreground mb-3 opacity-50" />
                    <p className="text-muted-foreground">No memories found</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {filteredMemories.map((memory) => (
                      <div key={`${memory.personaId}-${memory.originalIndex}`} className="p-3 bg-muted rounded-lg text-sm group">
                        <div className="flex items-center justify-between mb-1">
                          <Badge variant="outline" className="text-xs">{memory.personaName}</Badge>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground">
                              {safeParseDate(memory.timestamp).toLocaleString()}
                            </span>
                            <button
                              onClick={() => setDeleteMemoryConfirm({
                                personaId: memory.personaId,
                                index: memory.originalIndex,
                                content: safeContent(memory.content).slice(0, 50) + (safeContent(memory.content).length > 50 ? "..." : ""),
                              })}
                              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded text-destructive transition-opacity"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                        <p className="text-muted-foreground">{safeContent(memory.content)}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Dialogs */}
        <Dialog open={showNewFileDialog} onOpenChange={setShowNewFileDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Memory File</DialogTitle>
              <DialogDescription>Save information for the AI to reference later</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Filename</label>
                <Input value={newFileName} onChange={(e) => setNewFileName(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium">Content</label>
                <Textarea value={newFileContent} onChange={(e) => setNewFileContent(e.target.value)} rows={8} />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowNewFileDialog(false)}>Cancel</Button>
              <Button onClick={handleSaveNewFile}><Save className="w-4 h-4 mr-2" />Save</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={!!deleteConfirmFile} onOpenChange={() => setDeleteConfirmFile(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="text-destructive flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />Delete File?
              </DialogTitle>
              <DialogDescription>Delete &quot;{deleteConfirmFile}&quot;? This cannot be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteConfirmFile(null)}>Cancel</Button>
              <Button variant="destructive" onClick={handleDeleteFile}><Trash2 className="w-4 h-4 mr-2" />Delete</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={deleteAllConfirm === "files"} onOpenChange={() => setDeleteAllConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="text-destructive flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />Delete All Files?
              </DialogTitle>
              <DialogDescription>Delete all {files.length} memory files? This cannot be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteAllConfirm(null)}>Cancel</Button>
              <Button variant="destructive" onClick={handleDeleteAllFiles}><Trash2 className="w-4 h-4 mr-2" />Delete All</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={deleteAllConfirm === "conversations"} onOpenChange={() => setDeleteAllConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="text-destructive flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />Clear Memories?
              </DialogTitle>
              <DialogDescription>Clear {filteredMemories.length} conversation memories? This cannot be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteAllConfirm(null)}>Cancel</Button>
              <Button variant="destructive" onClick={handleDeleteAllConversationMemories}><Trash2 className="w-4 h-4 mr-2" />Clear</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={!!deleteMemoryConfirm} onOpenChange={() => setDeleteMemoryConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="text-destructive flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />Delete Memory?
              </DialogTitle>
              <DialogDescription>
                Delete this memory from {deleteMemoryConfirm ? personas.find(p => p.id === deleteMemoryConfirm.personaId)?.name : ""}?
                <div className="mt-3 p-2 bg-muted rounded text-sm italic">&quot;{deleteMemoryConfirm ? safeContent(deleteMemoryConfirm.content) : ""}&quot;</div>
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteMemoryConfirm(null)}>Cancel</Button>
              <Button variant="destructive" onClick={() => deleteMemoryConfirm && handleDeleteConversationMemory(deleteMemoryConfirm.personaId, deleteMemoryConfirm.index)}>
                <Trash2 className="w-4 h-4 mr-2" />Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}

// Memory write confirmation dialog
interface ConfirmationDialogProps {
  isOpen: boolean;
  filename: string;
  content?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function MemoryWriteConfirmation({ isOpen, filename, content, onConfirm, onCancel }: ConfirmationDialogProps) {
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-amber-500" />
            Confirm Memory Write
          </DialogTitle>
          <DialogDescription>The AI wants to save information to your memory folder.</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium">Filename</label>
            <div className="p-2 bg-muted rounded text-sm font-mono">{filename}</div>
          </div>
          {content && (
            <div>
              <label className="text-sm font-medium">Content Preview</label>
              <Textarea value={content.slice(0, 500) + (content.length > 500 ? "..." : "")} readOnly rows={6} className="font-mono text-sm" />
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}><X className="w-4 h-4 mr-2" />Cancel</Button>
          <Button onClick={onConfirm}><Check className="w-4 h-4 mr-2" />Confirm</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
