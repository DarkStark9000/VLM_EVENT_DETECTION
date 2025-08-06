#!/usr/bin/env python3
"""
Setup script for Visual Understanding Chat Assistant
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    directories = [
        "input_videos",
        "temp_videos", 
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created {directory}/")
    
    print("✅ Directories created!")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version compatible: {sys.version}")
    return True

def main():
    """Main setup function"""
    print("🚀 Visual Understanding Chat Assistant - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if not install_requirements():
        return 1
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run demo: python demo.py")
    print("2. Process video: python main.py --video /path/to/video.mp4")
    print("3. Interactive chat: python main.py --interactive")
    print("\nSee README.md for detailed usage instructions.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
