"""
Configuration management for Visual Understanding Chat Assistant
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """
    Configuration manager for the Visual Understanding Chat Assistant
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default config
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self.get_default_config()
        else:
            config = self.get_default_config()
            self.save_config(config)
            return config
    
    def save_config(self, config: Dict[str, Any] = None):
        """
        Save configuration to file
        """
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        """
        return {
            "models": {
                "vlm_model": "omni-research/Tarsier2-Recap-7b",
                "llm_model": "microsoft/Phi-3.5-mini-instruct",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
            },
            "video": {
                "max_duration_seconds": 120,
                "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
                "max_frames": 180,
                "fps": 1,
                "input_directory": "./input_videos",
                "temp_directory": "./temp_videos"
            },
            "database": {
                "chroma_db_path": "./chroma_db",
                "chat_history_db": "./chat_history.db",
                "collection_name": "video_segments"
            },
            "generation": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "context_window": 4096
            },
            "retrieval": {
                "top_k": 3,
                "similarity_threshold": 0.7,
                "max_context_length": 2000
            },
            "chat": {
                "max_history_turns": 10,
                "session_timeout_hours": 24,
                "default_user": "hackathon_user"
            },
            "hardware": {
                "device": "auto",  # "cuda", "cpu", or "auto"
                "mixed_precision": True,
                "max_memory_gb": 12
            },
            "logging": {
                "level": "INFO",
                "log_file": "./assistant.log",
                "console_output": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation (e.g., 'models.vlm_model')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        Set configuration value using dot notation and optionally save
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        if save:
            self.save_config()
    
    def validate_video_path(self, video_path: str) -> bool:
        """
        Validate if video path is acceptable
        """
        if not os.path.exists(video_path):
            return False
        
        # Check file extension
        supported_formats = self.get('video.supported_formats', [])
        file_ext = Path(video_path).suffix.lower()
        
        if file_ext not in supported_formats:
            return False
        
        # Check file size (basic check)
        try:
            file_size = os.path.getsize(video_path)
            max_size = self.get('video.max_size_mb', 500) * 1024 * 1024  # Default 500MB
            if file_size > max_size:
                print(f"Warning: Video file is large ({file_size / (1024*1024):.1f}MB)")
        except:
            pass
        
        return True
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get model-specific configuration
        """
        base_config = {
            "device": self.get('hardware.device', 'auto'),
            "mixed_precision": self.get('hardware.mixed_precision', True),
            "max_memory_gb": self.get('hardware.max_memory_gb', 12)
        }
        
        if model_type == 'vlm':
            base_config.update({
                "model_name": self.get('models.vlm_model'),
                "max_frames": self.get('video.max_frames', 180),
                "fps": self.get('video.fps', 1)
            })
        elif model_type == 'llm':
            base_config.update({
                "model_name": self.get('models.llm_model'),
                "max_new_tokens": self.get('generation.max_new_tokens', 512),
                "temperature": self.get('generation.temperature', 0.7),
                "context_window": self.get('generation.context_window', 4096)
            })
        elif model_type == 'embedding':
            base_config.update({
                "model_name": self.get('models.embedding_model')
            })
        
        return base_config
    
    def setup_directories(self):
        """
        Create necessary directories if they don't exist
        """
        directories = [
            self.get('video.input_directory'),
            self.get('video.temp_directory'),
            self.get('database.chroma_db_path'),
            os.path.dirname(self.get('database.chat_history_db')),
            os.path.dirname(self.get('logging.log_file'))
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    print(f"Created directory: {directory}")
                except Exception as e:
                    print(f"Warning: Could not create directory {directory}: {e}")

# Global configuration instance
config = Config()

# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key, default)

def set_config(key: str, value: Any, save: bool = True):
    """Set configuration value"""
    config.set(key, value, save)

def validate_video(video_path: str) -> bool:
    """Validate video path"""
    return config.validate_video_path(video_path)

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get model configuration"""
    return config.get_model_config(model_type)

def setup_directories():
    """Setup required directories"""
    config.setup_directories()

# Sample configuration file content for documentation
SAMPLE_CONFIG = """
{
  "models": {
    "vlm_model": "omni-research/Tarsier2-Recap-7b",
    "llm_model": "microsoft/Phi-3.5-mini-instruct", 
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
  },
  "video": {
    "max_duration_seconds": 120,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "max_frames": 180,
    "fps": 1,
    "input_directory": "./input_videos",
    "temp_directory": "./temp_videos"
  },
  "database": {
    "chroma_db_path": "./chroma_db",
    "chat_history_db": "./chat_history.db",
    "collection_name": "video_segments"
  },
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": true,
    "context_window": 4096
  },
  "hardware": {
    "device": "auto",
    "mixed_precision": true,
    "max_memory_gb": 12
  }
}
"""
