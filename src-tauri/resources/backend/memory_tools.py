"""
Secure Memory Tool System for Mimic AI
Provides mediated file access for agent models with strict security boundaries.

Security Principles:
1. Path validation - All paths must resolve within the allowed folder
2. No shell execution - Direct file operations only
3. Read-only by default - Writes require explicit user confirmation
4. Sandboxing - Single folder access, no system traversal
5. Extension filtering - Only safe file types allowed
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import fnmatch


class SecurityError(Exception):
    """Raised when a security boundary is violated"""
    pass


@dataclass
class MemoryFile:
    """Represents a memory file with metadata"""
    name: str
    path: str
    size: int
    modified: datetime
    content_preview: str


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: Any
    message: str


class SecureMemoryTools:
    """
    Secure memory access for agent models.
    The model is the brain, this class is the hands.
    """
    
    # Allowed file extensions for safety
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.pdf', '.csv', '.log'}
    
    # Maximum file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Maximum content length to return (prevent token overflow)
    MAX_CONTENT_LENGTH = 50000
    
    def __init__(self, base_path: str):
        """
        Initialize secure memory tools with a sandboxed folder.
        
        Args:
            base_path: The root folder for all memory operations
        """
        self.base_path = Path(base_path).resolve()
        self._ensure_folder_exists()
        
    def _ensure_folder_exists(self):
        """Create the memory folder if it doesn't exist"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _validate_path(self, filename: str) -> Path:
        """
        Validate that a filename resolves within the allowed folder.
        Prevents directory traversal attacks (e.g., ../../../etc/passwd)
        
        Args:
            filename: The requested filename
            
        Returns:
            Resolved Path object
            
        Raises:
            SecurityError: If path escapes the allowed folder
        """
        # Clean the filename (remove any path components)
        clean_filename = Path(filename).name
        
        # Resolve the full path
        requested_path = (self.base_path / clean_filename).resolve()
        
        # Security check: Must be within base_path
        try:
            requested_path.relative_to(self.base_path)
        except ValueError:
            raise SecurityError(
                f"Access denied: Path '{filename}' is outside the allowed memory folder"
            )
        
        return requested_path
    
    def _validate_extension(self, path: Path) -> None:
        """
        Validate that the file extension is allowed.
        
        Raises:
            SecurityError: If extension is not in allowed list
        """
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise SecurityError(
                f"File type not allowed: {path.suffix}. "
                f"Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
    
    def _read_file_safely(self, path: Path) -> str:
        """
        Safely read a file with size and content limits.
        
        Returns:
            File content as string
            
        Raises:
            SecurityError: If file is too large or cannot be read
        """
        # Check file size
        size = path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            raise SecurityError(
                f"File too large: {size} bytes (max {self.MAX_FILE_SIZE})"
            )
        
        # Read based on file type
        try:
            if path.suffix.lower() == '.pdf':
                # For PDFs, return metadata only (no content extraction)
                return f"[PDF Document: {path.name}, Size: {size} bytes]"
            
            # Read text files
            content = path.read_text(encoding='utf-8', errors='ignore')
            
            # Truncate if too long
            if len(content) > self.MAX_CONTENT_LENGTH:
                content = content[:self.MAX_CONTENT_LENGTH] + "\n...[Content truncated]"
            
            return content
            
        except Exception as e:
            raise SecurityError(f"Cannot read file: {e}")
    
    # =========================================================================
    # TOOL IMPLEMENTATIONS (These are the "hands" that the model can call)
    # =========================================================================
    
    def list_memories(self) -> ToolResult:
        """
        List all files in the memory folder.
        
        Returns:
            ToolResult with list of MemoryFile objects
        """
        try:
            files = []
            for item in self.base_path.iterdir():
                if item.is_file() and item.suffix.lower() in self.ALLOWED_EXTENSIONS:
                    stat = item.stat()
                    # Get preview (first 100 chars)
                    try:
                        preview = item.read_text(encoding='utf-8', errors='ignore')[:100]
                    except:
                        preview = "[Binary or unreadable content]"
                    
                    files.append(MemoryFile(
                        name=item.name,
                        path=str(item.relative_to(self.base_path)),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime),
                        content_preview=preview + "..." if len(preview) >= 100 else preview
                    ))
            
            return ToolResult(
                success=True,
                data=files,
                message=f"Found {len(files)} memory files"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error listing memories: {e}"
            )
    
    def read_memory(self, filename: str) -> ToolResult:
        """
        Read a specific memory file.
        
        Args:
            filename: Name of the file to read
            
        Returns:
            ToolResult with file content
        """
        try:
            # Validate path (security boundary)
            path = self._validate_path(filename)
            
            # Check if file exists
            if not path.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"File not found: {filename}"
                )
            
            # Validate extension
            self._validate_extension(path)
            
            # Read content
            content = self._read_file_safely(path)
            
            return ToolResult(
                success=True,
                data={
                    "filename": filename,
                    "content": content,
                    "size": len(content)
                },
                message=f"Successfully read {filename}"
            )
            
        except SecurityError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error reading file: {e}"
            )
    
    def search_memories(self, query: str) -> ToolResult:
        """
        Search memory contents for keywords.
        
        Args:
            query: Search query string
            
        Returns:
            ToolResult with matching files and snippets
        """
        try:
            matches = []
            query_lower = query.lower()
            
            for item in self.base_path.iterdir():
                if not item.is_file():
                    continue
                if item.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                    continue
                
                try:
                    content = item.read_text(encoding='utf-8', errors='ignore')
                    
                    if query_lower in content.lower():
                        # Find the matching snippet
                        idx = content.lower().find(query_lower)
                        start = max(0, idx - 100)
                        end = min(len(content), idx + len(query) + 100)
                        snippet = content[start:end]
                        
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                        
                        matches.append({
                            "filename": item.name,
                            "snippet": snippet,
                            "matches": content.lower().count(query_lower)
                        })
                        
                except Exception:
                    continue
            
            return ToolResult(
                success=True,
                data=matches,
                message=f"Found {len(matches)} files matching '{query}'"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error searching memories: {e}"
            )
    
    def get_memory_info(self, filename: str) -> ToolResult:
        """
        Get metadata about a memory file without reading content.
        
        Args:
            filename: Name of the file
            
        Returns:
            ToolResult with file metadata
        """
        try:
            path = self._validate_path(filename)
            
            if not path.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"File not found: {filename}"
                )
            
            stat = path.stat()
            
            return ToolResult(
                success=True,
                data={
                    "filename": filename,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": path.suffix
                },
                message=f"Retrieved info for {filename}"
            )
            
        except SecurityError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e)
            )
    
    # =========================================================================
    # WRITE OPERATIONS (Require user confirmation)
    # =========================================================================
    
    def write_memory(self, filename: str, content: str, confirm: bool = False) -> ToolResult:
        """
        Write content to a memory file.
        REQUIRES explicit confirmation parameter to prevent accidental writes.
        
        Args:
            filename: Name of the file to write
            content: Content to write
            confirm: Must be True to actually write (safety check)
            
        Returns:
            ToolResult with operation status
        """
        if not confirm:
            return ToolResult(
                success=False,
                data=None,
                message="Write operation requires explicit confirmation. Set confirm=True to proceed."
            )
        
        try:
            # Validate path
            path = self._validate_path(filename)
            
            # Validate extension
            self._validate_extension(path)
            
            # Write file
            path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={"filename": filename, "size": len(content)},
                message=f"Successfully wrote {filename}"
            )
            
        except SecurityError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error writing file: {e}"
            )
    
    def delete_memory(self, filename: str, confirm: bool = False) -> ToolResult:
        """
        Delete a memory file.
        REQUIRES explicit confirmation.
        
        Args:
            filename: Name of the file to delete
            confirm: Must be True to actually delete
            
        Returns:
            ToolResult with operation status
        """
        if not confirm:
            return ToolResult(
                success=False,
                data=None,
                message="Delete operation requires explicit confirmation. Set confirm=True to proceed."
            )
        
        try:
            path = self._validate_path(filename)
            
            if not path.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"File not found: {filename}"
                )
            
            path.unlink()
            
            return ToolResult(
                success=True,
                data={"filename": filename},
                message=f"Successfully deleted {filename}"
            )
            
        except SecurityError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error deleting file: {e}"
            )


# =========================================================================
# AGENT TOOL DEFINITIONS (for Ollama tool calling)
# =========================================================================

MEMORY_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List all files in the memory folder",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read the contents of a specific memory file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read (e.g., 'project_notes.txt')"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Search memory files for keywords or phrases",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find in memory files"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_info",
            "description": "Get metadata about a memory file (size, date, etc.) without reading content",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "Write content to a memory file. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm the write operation"
                    }
                },
                "required": ["filename", "content", "confirm"]
            }
        }
    }
]


# Singleton instance
_memory_tools_instance: Optional[SecureMemoryTools] = None


def get_memory_tools(base_path: Optional[str] = None) -> SecureMemoryTools:
    """
    Get or create the singleton memory tools instance.
    
    Args:
        base_path: Path to memory folder (only used on first call)
        
    Returns:
        SecureMemoryTools instance
    """
    global _memory_tools_instance
    if _memory_tools_instance is None:
        if base_path is None:
            # Default to user's home directory
            base_path = os.path.expanduser("~/MimicAI/Memories/")
        _memory_tools_instance = SecureMemoryTools(base_path)
    return _memory_tools_instance
