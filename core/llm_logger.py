"""
LLM Interaction Logger

This module provides functionality for logging all interactions with LLM APIs,
including prompts, responses, and contracts at each step of the pipeline.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

class LLMLogger:
    """
    Logger for tracking all LLM interactions in the YouTube Shorts generation pipeline.
    Saves detailed logs to files for analysis and debugging.
    """
    
    def __init__(self, log_dir: Union[str, Path] = "logs"):
        """
        Initialize the LLM logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a session ID based on timestamp
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize log entries
        self.logs = []
        
        # Create a summary log file
        self.summary_file = self.session_dir / "session_summary.json"
        with open(self.summary_file, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "logs": []
            }, f, indent=2)
    
    def log_interaction(self, 
                       stage: str, 
                       prompt: str, 
                       response: Any, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an interaction with an LLM.
        
        Args:
            stage: Pipeline stage (e.g., "narrative_planning", "clip_selection")
            prompt: The prompt sent to the LLM
            response: The response received from the LLM
            metadata: Additional metadata about the interaction
            
        Returns:
            Path to the created log file
        """
        # Create a log entry
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        
        # Add to logs list
        self.logs.append(log_entry)
        
        # Create a file for this specific interaction
        log_filename = f"{stage}_{len(self.logs):03d}.json"
        log_file = self.session_dir / log_filename
        
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)
        
        # Update summary file
        with open(self.summary_file, "r") as f:
            summary = json.load(f)
        
        summary["logs"].append({
            "timestamp": log_entry["timestamp"],
            "stage": stage,
            "file": log_filename
        })
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return str(log_file)
    
    def log_contract(self, 
                    stage: str, 
                    input_contract: Dict[str, Any],
                    output_contract: Dict[str, Any],
                    validation_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a contract between pipeline components.
        
        Args:
            stage: Pipeline stage (e.g., "narrative_planning", "clip_selection")
            input_contract: The expected input schema/format
            output_contract: The expected output schema/format
            validation_result: Result of contract validation, if applicable
            
        Returns:
            Path to the created log file
        """
        # Create a contract entry
        contract_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "input_contract": input_contract,
            "output_contract": output_contract,
            "validation_result": validation_result
        }
        
        # Create a file for this contract
        contract_filename = f"{stage}_contract.json"
        contract_file = self.session_dir / contract_filename
        
        with open(contract_file, "w") as f:
            json.dump(contract_entry, f, indent=2)
        
        # Update summary file
        with open(self.summary_file, "r") as f:
            summary = json.load(f)
        
        if "contracts" not in summary:
            summary["contracts"] = []
        
        summary["contracts"].append({
            "timestamp": contract_entry["timestamp"],
            "stage": stage,
            "file": contract_filename
        })
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return str(contract_file)
    
    def close(self) -> None:
        """
        Finalize logging for this session.
        """
        # Update summary with end time
        with open(self.summary_file, "r") as f:
            summary = json.load(f)
        
        summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary["total_interactions"] = len(self.logs)
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

# Global instance for convenient access
llm_logger = LLMLogger()

def get_llm_logger() -> LLMLogger:
    """Get the global LLM logger instance."""
    return llm_logger
