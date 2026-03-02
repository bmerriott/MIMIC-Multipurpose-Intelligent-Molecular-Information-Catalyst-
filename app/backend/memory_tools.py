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
    
    Folder Structure (Per-Persona - including 'default'):
    Each persona (including the default one) gets its own isolated folder:
    - {persona_id}/user_files/     : User-created files (notes, documents, etc.)
    - {persona_id}/conversations/  : Auto-saved conversation history per persona  
    - {persona_id}/history.json    : Full conversation history
    
    Example:
    - default/user_files/          # Default persona's files
    - default/history.json         # Default persona's conversation history
    - persona_123/user_files/      # Custom persona's files
    - persona_123/history.json     # Custom persona's conversation history
    
    NO file size limits - storage is local and user-managed.
    """
    
    # Allowed file extensions for safety
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.pdf', '.csv', '.log'}
    
    # Maximum content length to return (prevent token overflow)
    MAX_CONTENT_LENGTH = 50000
    
    # NOTE: No file size limit enforced - users can upload any size file
    # Storage is local and user-managed
    
    def __init__(self, base_path: str):
        """
        Initialize secure memory tools with a sandboxed folder.
        
        Args:
            base_path: The root folder for all memory operations
        """
        self.base_path = Path(base_path).resolve()
        # Create default folder for backward compatibility
        self._get_persona_paths("default")
        
    def _get_persona_paths(self, persona_id: str = "default") -> Dict[str, Path]:
        """
        Get paths for a specific persona's memory folders.
        
        Args:
            persona_id: The persona identifier (default for global memories)
            
        Returns:
            Dictionary with paths for user_files, conversations, and history
        """
        persona_path = self.base_path / persona_id
        paths = {
            "base": persona_path,
            "user_files": persona_path / "user_files",
            "conversations": persona_path / "conversations",
            "history": persona_path / "history.json"
        }
        
        # Ensure folders exist
        persona_path.mkdir(parents=True, exist_ok=True)
        paths["user_files"].mkdir(exist_ok=True)
        paths["conversations"].mkdir(exist_ok=True)
        
        return paths
    
    def _get_folder_path(self, folder: str = "user_files", persona_id: str = "default") -> Path:
        """
        Get the appropriate folder path based on folder type and persona.
        
        Args:
            folder: 'user_files' or 'conversations'
            persona_id: The persona identifier
            
        Returns:
            Path object for the requested folder
        """
        paths = self._get_persona_paths(persona_id)
        if folder == "conversations":
            return paths["conversations"]
        return paths["user_files"]
        
    def _validate_path(self, filename: str, persona_id: str = "default") -> Path:
        """
        Validate that a filename resolves within the allowed folder.
        Prevents directory traversal attacks (e.g., ../../../etc/passwd)
        
        Args:
            filename: The requested filename
            persona_id: The persona identifier
            
        Returns:
            Resolved Path object
            
        Raises:
            SecurityError: If path escapes the allowed folder
        """
        paths = self._get_persona_paths(persona_id)
        
        # Clean the filename (remove any path components)
        clean_filename = Path(filename).name
        
        # Try user_files first, then conversations
        for folder in ["user_files", "conversations"]:
            requested_path = (paths[folder] / clean_filename).resolve()
            try:
                requested_path.relative_to(paths[folder])
                return requested_path
            except ValueError:
                continue
        
        # If neither, check against base persona path
        requested_path = (paths["base"] / clean_filename).resolve()
        try:
            requested_path.relative_to(paths["base"])
            return requested_path
        except ValueError:
            raise SecurityError(
                f"Access denied: Path '{filename}' is outside the allowed memory folder"
            )
    
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
    # CONVERSATION HISTORY MANAGEMENT
    # =========================================================================
    
    def save_conversation_message(self, persona_id: str, role: str, content: str, 
                                   message_type: str = "text", metadata: Dict = None) -> ToolResult:
        """
        Save a conversation message to the persona's history file.
        
        Args:
            persona_id: The persona identifier
            role: 'user' or 'assistant'
            content: The message content
            message_type: Type of message (text, image, file, etc.)
            metadata: Additional metadata (timestamps, etc.)
            
        Returns:
            ToolResult with operation status
        """
        try:
            paths = self._get_persona_paths(persona_id)
            history_file = paths["history"]
            
            # Load existing history
            history = []
            if history_file.exists():
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except:
                    history = []
            
            # Add new message
            message = {
                "role": role,
                "content": content,
                "type": message_type,
                "timestamp": metadata.get("timestamp") if metadata else datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            history.append(message)
            
            # Save back to file
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            return ToolResult(
                success=True,
                data={"message_count": len(history)},
                message=f"Saved {role} message to {persona_id} history"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error saving conversation: {e}"
            )
    
    def get_conversation_history(self, persona_id: str, limit: int = None) -> ToolResult:
        """
        Get the full conversation history for a persona.
        
        Args:
            persona_id: The persona identifier
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            ToolResult with conversation history
        """
        try:
            paths = self._get_persona_paths(persona_id)
            history_file = paths["history"]
            
            if not history_file.exists():
                return ToolResult(
                    success=True,
                    data=[],
                    message="No conversation history found"
                )
            
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if limit and len(history) > limit:
                history = history[-limit:]
            
            return ToolResult(
                success=True,
                data=history,
                message=f"Retrieved {len(history)} messages from {persona_id} history"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error reading conversation history: {e}"
            )
    
    def search_conversation_history(self, persona_id: str, query: str) -> ToolResult:
        """
        Search conversation history for specific content.
        
        Args:
            persona_id: The persona identifier
            query: Search query string
            
        Returns:
            ToolResult with matching messages
        """
        try:
            result = self.get_conversation_history(persona_id)
            if not result.success:
                return result
            
            history = result.data
            query_lower = query.lower()
            
            matches = []
            for msg in history:
                content = msg.get("content", "")
                if query_lower in content.lower():
                    matches.append(msg)
            
            return ToolResult(
                success=True,
                data=matches,
                message=f"Found {len(matches)} matching messages"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error searching conversation history: {e}"
            )
    
    def clear_conversation_history(self, persona_id: str, confirm: bool = False) -> ToolResult:
        """
        Clear all conversation history for a persona.
        
        Args:
            persona_id: The persona identifier
            confirm: Must be True to actually clear
            
        Returns:
            ToolResult with operation status
        """
        if not confirm:
            return ToolResult(
                success=False,
                data=None,
                message="Clear operation requires explicit confirmation"
            )
        
        try:
            paths = self._get_persona_paths(persona_id)
            history_file = paths["history"]
            
            if history_file.exists():
                history_file.unlink()
            
            return ToolResult(
                success=True,
                data=None,
                message=f"Cleared conversation history for {persona_id}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error clearing history: {e}"
            )
    
    # =========================================================================
    # TOOL IMPLEMENTATIONS (These are the "hands" that the model can call)
    # =========================================================================
    
    def list_memories(self, folder: str = None, persona_id: str = "default") -> ToolResult:
        """
        List all files in the memory folders for a persona.
        
        Args:
            folder: 'user_files', 'conversations', or None for both
            persona_id: The persona identifier
            
        Returns:
            ToolResult with list of MemoryFile objects
        """
        try:
            paths = self._get_persona_paths(persona_id)
            files = []
            
            folders_to_scan = []
            if folder is None or folder == "all":
                folders_to_scan = [("user_files", paths["user_files"]), 
                                   ("conversations", paths["conversations"])]
            elif folder == "conversations":
                folders_to_scan = [("conversations", paths["conversations"])]
            else:
                folders_to_scan = [("user_files", paths["user_files"])]
            
            for folder_name, folder_path in folders_to_scan:
                if not folder_path.exists():
                    continue
                for item in folder_path.iterdir():
                    if item.is_file() and item.suffix.lower() in self.ALLOWED_EXTENSIONS:
                        stat = item.stat()
                        # Get preview (first 100 chars)
                        try:
                            preview = item.read_text(encoding='utf-8', errors='ignore')[:100]
                        except:
                            preview = "[Binary or unreadable content]"
                        
                        files.append(MemoryFile(
                            name=item.name,
                            path=f"{folder_name}/{item.name}",
                            size=stat.st_size,
                            modified=datetime.fromtimestamp(stat.st_mtime),
                            content_preview=preview + "..." if len(preview) >= 100 else preview
                        ))
            
            return ToolResult(
                success=True,
                data=files,
                message=f"Found {len(files)} memory files for {persona_id}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                message=f"Error listing memories: {e}"
            )
    
    def read_memory(self, filename: str, persona_id: str = "default") -> ToolResult:
        """
        Read a specific memory file for a persona.
        
        Args:
            filename: Name of the file to read (can include folder prefix like "user_files/" or "conversations/")
            persona_id: The persona identifier
            
        Returns:
            ToolResult with file content
        """
        try:
            # Determine folder from path prefix
            folder = "user_files"
            clean_filename = filename
            
            if filename.startswith("user_files/"):
                folder = "user_files"
                clean_filename = filename[11:]  # Remove prefix
            elif filename.startswith("conversations/"):
                folder = "conversations"
                clean_filename = filename[14:]  # Remove prefix
            
            paths = self._get_persona_paths(persona_id)
            folder_path = paths[folder]
            
            # Validate path (security boundary)
            clean_filename = Path(clean_filename).name  # Strip any remaining path
            path = (folder_path / clean_filename).resolve()
            
            # Security check: Must be within folder
            try:
                path.relative_to(folder_path)
            except ValueError:
                raise SecurityError(f"Access denied: Path '{filename}' is outside the allowed folder")
            
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
    
    def search_memories(self, query: str, folder: str = None, persona_id: str = "default") -> ToolResult:
        """
        Search memory contents for keywords.
        
        Args:
            query: Search query string
            folder: 'user_files', 'conversations', or None for both
            persona_id: The persona identifier
            
        Returns:
            ToolResult with matching files and snippets
        """
        try:
            matches = []
            query_lower = query.lower()
            
            paths = self._get_persona_paths(persona_id)
            
            # Determine folders to search
            folders_to_search = []
            if folder is None or folder == "all":
                folders_to_search = [("user_files", paths["user_files"]), 
                                     ("conversations", paths["conversations"])]
            elif folder == "conversations":
                folders_to_search = [("conversations", paths["conversations"])]
            else:
                folders_to_search = [("user_files", paths["user_files"])]
            
            for folder_name, folder_path in folders_to_search:
                if not folder_path.exists():
                    continue
                    
                for item in folder_path.iterdir():
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
                                "filename": f"{folder_name}/{item.name}",
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
    
    def get_memory_info(self, filename: str, persona_id: str = "default") -> ToolResult:
        """
        Get metadata about a memory file without reading content.
        
        Args:
            filename: Name of the file (can include folder prefix)
            persona_id: The persona identifier
            
        Returns:
            ToolResult with file metadata
        """
        try:
            # Determine folder from path prefix
            folder = "user_files"
            clean_filename = filename
            
            if filename.startswith("user_files/"):
                folder = "user_files"
                clean_filename = filename[11:]
            elif filename.startswith("conversations/"):
                folder = "conversations"
                clean_filename = filename[14:]
            
            paths = self._get_persona_paths(persona_id)
            folder_path = paths[folder]
            clean_filename = Path(clean_filename).name
            path = folder_path / clean_filename
            
            # Security check
            try:
                path.relative_to(folder_path)
            except ValueError:
                raise SecurityError(f"Access denied: Invalid path")
            
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
    
    def write_memory(self, filename: str, content: str, confirm: bool = False, 
                     folder: str = "user_files", persona_id: str = "default") -> ToolResult:
        """
        Write content to a memory file.
        REQUIRES explicit confirmation parameter to prevent accidental writes.
        
        Args:
            filename: Name of the file to write (without folder prefix)
            content: Content to write
            confirm: Must be True to actually write (safety check)
            folder: 'user_files' or 'conversations'
            persona_id: The persona identifier
            
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
            paths = self._get_persona_paths(persona_id)
            folder_path = paths[folder]
            
            # Clean filename and build path
            clean_filename = Path(filename).name
            path = folder_path / clean_filename
            
            # Security check
            try:
                path.relative_to(folder_path)
            except ValueError:
                raise SecurityError(f"Access denied: Invalid path")
            
            # Validate extension
            self._validate_extension(path)
            
            # Ensure folder exists
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Write file
            path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                data={"filename": f"{folder}/{clean_filename}", "size": len(content)},
                message=f"Successfully wrote {folder}/{clean_filename}"
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
    
    def delete_memory(self, filename: str, confirm: bool = False, persona_id: str = "default") -> ToolResult:
        """
        Delete a memory file.
        REQUIRES explicit confirmation.
        
        Args:
            filename: Name of the file to delete (can include folder prefix)
            confirm: Must be True to actually delete
            persona_id: The persona identifier
            
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
            # Determine folder from path prefix
            folder = "user_files"
            clean_filename = filename
            
            if filename.startswith("user_files/"):
                folder = "user_files"
                clean_filename = filename[11:]
            elif filename.startswith("conversations/"):
                folder = "conversations"
                clean_filename = filename[14:]
            
            paths = self._get_persona_paths(persona_id)
            folder_path = paths[folder]
            clean_filename = Path(clean_filename).name
            path = folder_path / clean_filename
            
            # Security check
            try:
                path.relative_to(folder_path)
            except ValueError:
                raise SecurityError(f"Access denied: Invalid path")
            
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
            "description": "List all files in the memory folder for the current persona",
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
            "name": "get_conversation_history",
            "description": "Get the full conversation history with the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to retrieve (default: all)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_conversation_history",
            "description": "Search conversation history for specific content",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find in conversation history"
                    }
                },
                "required": ["query"]
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
