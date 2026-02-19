import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Paperclip, FileText, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import { toast } from "sonner";

interface FileAttachmentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAttach: (name: string, content: string) => void;
}

// Supported file types
const SUPPORTED_TYPES: Record<string, string[]> = {
  'text/plain': ['.txt', '.md', '.log'],
  'text/markdown': ['.md', '.markdown'],
  'application/json': ['.json'],
  'text/csv': ['.csv'],
  'text/html': ['.html', '.htm'],
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/msword': ['.doc'],
  'application/rtf': ['.rtf'],
  'text/css': ['.css'],
  'text/javascript': ['.js', '.jsx'],
  'text/typescript': ['.ts', '.tsx'],
  'text/python': ['.py'],
  'text/x-python': ['.py'],
  'text/java': ['.java'],
  'text/x-c': ['.c', '.cpp', '.h'],
  'text/x-c++': ['.cpp', '.cc', '.hpp'],
  'text/rust': ['.rs'],
  'text/go': ['.go'],
  'text/ruby': ['.rb'],
  'text/php': ['.php'],
  'text/sql': ['.sql'],
  'text/yaml': ['.yaml', '.yml'],
  'text/xml': ['.xml'],
  'text/x-sh': ['.sh', '.bash'],
  'application/x-httpd-php': ['.php'],
};

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB max
const MAX_CHARS = 50000; // Max characters to extract

export function FileAttachmentModal({ isOpen, onClose, onAttach }: FileAttachmentModalProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isFileSupported = (file: File): boolean => {
    // Check by MIME type
    if (SUPPORTED_TYPES[file.type]) return true;
    
    // Check by extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    return Object.values(SUPPORTED_TYPES).flat().includes(ext);
  };

  const extractTextFromPDF = async (file: File): Promise<string> => {
    // For PDF extraction, we'll use a simple approach
    // In a full implementation, you'd use pdf.js
    // For now, we'll read as text and try to extract readable content
    const text = await file.text();
    // Basic cleanup - remove non-printable characters
    return text.replace(/[^\x20-\x7E\n\r\t]/g, ' ').slice(0, MAX_CHARS);
  };

  const extractTextFromDOCX = async (file: File): Promise<string> => {
    // DOCX is a zip file with XML inside
    // For a full implementation, use mammoth.js
    // For now, read as text and extract readable strings
    const text = await file.text();
    // Extract text between XML tags
    const extracted = text
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, MAX_CHARS);
    return extracted;
  };

  const processFile = async (file: File): Promise<string> => {
    if (file.size > MAX_FILE_SIZE) {
      throw new Error(`File too large. Max size is 5MB.`);
    }

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();

    try {
      // PDF files
      if (file.type === 'application/pdf' || ext === '.pdf') {
        return await extractTextFromPDF(file);
      }

      // Word documents
      if (ext === '.docx' || ext === '.doc') {
        return await extractTextFromDOCX(file);
      }

      // Text-based files - read directly
      const text = await file.text();
      
      // Truncate if too long
      if (text.length > MAX_CHARS) {
        toast.info(`File truncated to ${MAX_CHARS} characters`);
        return text.slice(0, MAX_CHARS) + '\n\n[File truncated...]';
      }

      return text;
    } catch (error) {
      throw new Error(`Failed to read file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleFile = async (file: File) => {
    if (!isFileSupported(file)) {
      toast.error(`Unsupported file type: ${file.name}`);
      return;
    }

    setIsProcessing(true);
    try {
      const content = await processFile(file);
      onAttach(file.name, content);
      toast.success(`File attached: ${file.name}`);
      onClose();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to process file');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }, [onAttach, onClose]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="bg-background border rounded-xl overflow-hidden max-w-md w-full shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-muted/50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Paperclip className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold">Attach File</h3>
                  <p className="text-xs text-muted-foreground">
                    Upload a document for the AI to read
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={onClose} disabled={isProcessing}>
                <X className="w-5 h-5" />
              </Button>
            </div>

            {/* Drop Zone */}
            <div className="p-4">
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
                  isDragging
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/25 hover:border-muted-foreground/50'
                } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
                onClick={() => !isProcessing && fileInputRef.current?.click()}
              >
                {isProcessing ? (
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="w-10 h-10 text-primary animate-spin" />
                    <p className="text-muted-foreground">Processing file...</p>
                  </div>
                ) : (
                  <>
                    <FileText className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-sm font-medium mb-1">
                      Drop file here or click to browse
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Max 5MB • Text extracted and attached
                    </p>
                  </>
                )}
              </div>

              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                className="hidden"
                accept=".txt,.md,.pdf,.docx,.doc,.json,.csv,.html,.css,.js,.ts,.py,.java,.cpp,.c,.h,.rs,.go,.rb,.php,.sql,.yaml,.yml,.xml"
              />

              {/* Supported formats */}
              <div className="mt-4">
                <p className="text-xs text-muted-foreground mb-2">Supported formats:</p>
                <div className="flex flex-wrap gap-1">
                  {['TXT', 'PDF', 'DOCX', 'MD', 'JSON', 'CSV', 'Code files'].map((format) => (
                    <span
                      key={format}
                      className="px-2 py-0.5 bg-muted rounded text-[10px] text-muted-foreground"
                    >
                      {format}
                    </span>
                  ))}
                </div>
              </div>

              {/* Privacy note */}
              <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground">
                  <strong className="text-foreground">Privacy:</strong> File content is read once and 
                  included in the conversation. Files are not stored permanently. The AI will have 
                  read-only access to analyze and reference the content.
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
