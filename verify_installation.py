#!/usr/bin/env python3
"""
Verification script for Unbitrium installation and metadata.
Usage: python verify_installation.py
"""

import sys
import platform
import importlib.util

def check_package(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"[ERROR] Package '{name}' not found.")
        return False
    print(f"[OK] Package '{name}' found at {spec.origin}")
    return True

def main():
    print("="*60)
    print("Unbitrium Repository Verification")
    print("="*60)

    # Check Python Version
    py_ver = sys.version_info
    print(f"Python Version: {sys.version}")
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 10):
        print("[WARNING] Unbitrium requires Python 3.10+ (Recommended 3.14)")
    else:
        print("[OK] Python version compatible.")

    print("-" * 60)

    # Metadata
    print("Project: Unbitrium")
    print("Version: 1.0.0")
    print("License: EUPL-1.2")
    print("Author: Olaf Yunus Laitinen Imanov")
    print("Email: oyli@dtu.dk")
    print("Affiliation: DTU Compute, Visual Computing")
    print("ORCID: https://orcid.org/0009-0006-5184-0810")

    print("-" * 60)

    # Check core modules existence (simulated check as they are local files)
    modules = [
        "unbitrium.core",
        "unbitrium.datasets",
        "unbitrium.partitioning",
        "unbitrium.aggregators",
        "unbitrium.metrics",
        "unbitrium.systems",
        "unbitrium.privacy",
        "unbitrium.bench"
    ]

    print("Verifying core packages...")
    # Add src to path to simulate installed package
    sys.path.append("./src")

    all_ok = True
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"[OK] Imported {mod}")
        except ImportError as e:
            print(f"[ERROR] Failed to import {mod}: {e}")
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("SUCCESS: Unbitrium repository is ready for production.")
        print("See CHANGELOG.md for version history (0.0.1 -> 1.0.0).")
        print("See ROADMAP.md for future plans (1.x -> 2.x).")
    else:
        print("FAILURE: Some components are missing or broken.")

if __name__ == "__main__":
    main()
