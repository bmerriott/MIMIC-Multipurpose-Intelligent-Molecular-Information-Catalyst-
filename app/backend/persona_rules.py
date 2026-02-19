"""
Persona Rules System
Manages persistent rules.md files for each persona
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class PersonaRulesManager:
    """Manages rules.md files for personas"""
    
    RULES_FILENAME = "rules.md"
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()
        self._ensure_base_folder()
    
    def _ensure_base_folder(self):
        """Create the personas folder if it doesn't exist"""
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_persona_folder(self, persona_id: str) -> Path:
        """Get the folder for a specific persona"""
        # Sanitize persona ID for filesystem safety
        safe_id = "".join(c for c in persona_id if c.isalnum() or c in ('_', '-')).rstrip()
        folder = self.base_path / safe_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    
    def _get_rules_path(self, persona_id: str) -> Path:
        """Get the path to a persona's rules.md file"""
        return self._get_persona_folder(persona_id) / self.RULES_FILENAME
    
    def get_rules(self, persona_id: str) -> Optional[str]:
        """
        Get the rules.md content for a persona.
        Returns None if no rules file exists.
        """
        rules_path = self._get_rules_path(persona_id)
        if not rules_path.exists():
            return None
        try:
            return rules_path.read_text(encoding='utf-8')
        except Exception:
            return None
    
    def save_rules(self, persona_id: str, content: str) -> bool:
        """
        Save rules.md content for a persona.
        Returns True on success.
        """
        try:
            rules_path = self._get_rules_path(persona_id)
            rules_path.write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False
    
    def update_rules_from_persona(self, persona_id: str, persona_data: Dict[str, Any]) -> bool:
        """
        Generate or update rules.md from persona configuration.
        This bakes the personality and settings into the rules file.
        """
        try:
            # Build rules content
            lines = []
            
            # Header
            lines.append(f"# {persona_data.get('name', 'Assistant')} - System Rules")
            lines.append("")
            lines.append(f"Generated: {datetime.now().isoformat()}")
            lines.append("")
            
            # Core identity
            lines.append("## Core Identity")
            lines.append("")
            lines.append(f"You are {persona_data.get('name', 'the Assistant')}, an AI Digital Assistant.")
            lines.append("")
            
            # Personality
            personality = persona_data.get('personality_prompt', '')
            if personality:
                lines.append("## Personality")
                lines.append("")
                lines.append(personality)
                lines.append("")
            
            # Background/Description
            description = persona_data.get('description', '')
            if description and description != personality:
                lines.append("## Background")
                lines.append("")
                lines.append(description)
                lines.append("")
            
            # Behavior rules
            lines.append("## Behavior Rules")
            lines.append("")
            lines.append("- Respond naturally without referencing technical modes or input methods")
            lines.append("- Be helpful, engaging, and genuine")
            lines.append("- Stay in character while being respectful")
            lines.append("- No stage directions or action markers (* or [])")
            lines.append("- Never promote hate speech, violence, or bigotry")
            lines.append("")
            
            # Capabilities
            lines.append("## Capabilities")
            lines.append("")
            lines.append("You can:")
            lines.append("- Answer questions using real-time web search results when provided")
            lines.append("- Read and reference files attached by the user")
            lines.append("- Access a personal memory folder for long-term recall")
            lines.append("- Respond via text and voice synthesis")
            lines.append("")
            lines.append("If asked about your capabilities, explain you run on Mimic AI,")
            lines.append("a multi-modal system combining local LLMs with web search,")
            lines.append("file reading, vision, and voice synthesis.")
            lines.append("")
            
            # Web search guidance
            lines.append("## Using Web Search Data")
            lines.append("")
            lines.append("When search results are provided in your context, answer directly")
            lines.append("using that information. Include relevant source URLs.")
            lines.append("If results are insufficient, say so clearly.")
            lines.append("")
            
            # Memory folder access
            lines.append("## Memory Access")
            lines.append("")
            lines.append("You have read access to files in your memory folder.")
            lines.append("You can request to save notes to your memory folder.")
            lines.append("All file writes require explicit user confirmation.")
            lines.append("")
            
            # Join and save
            content = "\n".join(lines)
            return self.save_rules(persona_id, content)
            
        except Exception:
            return False
    
    def delete_persona_rules(self, persona_id: str) -> bool:
        """Delete a persona's rules file and folder"""
        try:
            folder = self._get_persona_folder(persona_id)
            rules_path = folder / self.RULES_FILENAME
            if rules_path.exists():
                rules_path.unlink()
            # Try to remove empty folder
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
            return True
        except Exception:
            return False


# Singleton instance
_rules_manager: Optional[PersonaRulesManager] = None


def get_persona_rules_manager(base_path: Optional[str] = None) -> PersonaRulesManager:
    """Get or create the singleton rules manager"""
    global _rules_manager
    if _rules_manager is None:
        if base_path is None:
            base_path = os.path.expanduser("~/MimicAI/Personas/")
        _rules_manager = PersonaRulesManager(base_path)
    return _rules_manager
