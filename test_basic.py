#!/usr/bin/env python3
"""
Basic test without Flask to check Python environment
"""
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current directory:", os.getcwd())
print("Files in current directory:")
for f in os.listdir('.'):
    print(f"  {f}")

# Test basic imports
try:
    import json
    print("✓ json module works")
except ImportError as e:
    print("✗ json import failed:", e)

try:
    import pathlib
    print("✓ pathlib module works")
except ImportError as e:
    print("✗ pathlib import failed:", e)

# Test dataset directory
dataset_dir = "Dataset/data/data"
if os.path.exists(dataset_dir):
    print(f"✓ Dataset directory exists: {dataset_dir}")
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"  Found {len(subdirs)} subdirectories (job categories)")
    for subdir in subdirs[:5]:  # Show first 5
        pdf_count = len([f for f in os.listdir(os.path.join(dataset_dir, subdir)) if f.endswith('.pdf')])
        print(f"    {subdir}: {pdf_count} PDFs")
else:
    print(f"✗ Dataset directory not found: {dataset_dir}")

print("\nTest completed successfully!")
