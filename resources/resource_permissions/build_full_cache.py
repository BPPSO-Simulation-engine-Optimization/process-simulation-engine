#!/usr/bin/env python3
"""
Build Full OrdinoR Model Cache

This script processes the FULL event log to build the OrdinoR organizational model.
WARNING: This takes 15-60 minutes depending on your system.

Usage:
    PYTHONPATH=. python3 resources/build_full_cache.py
    
The resulting cache file can then be reused instantly via ResourceAllocator's cache_path parameter.
"""

import os
import time
from resources.resource_allocation import ResourceAllocator

LOG_PATH = "eventlog.xes.gz"
CACHE_FILE = "ordinor_model_full.pkl"

def main():
    print("="*70)
    print(" OrdinoR Full Model Cache Builder")
    print("="*70)
    print()
    print("⚠️  WARNING: This will process the FULL event log")
    print("   Estimated time: 15-60 minutes (depending on CPU)")
    print()
    print(f"   Input:  {LOG_PATH}")
    print(f"   Output: {CACHE_FILE}")
    print()
    
    response = input("Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    print("\n" + "="*70)
    print("Starting Full Log Processing...")
    print("="*70 + "\n")
    
    # Clean existing cache
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"[+] Removed existing cache: {CACHE_FILE}\n")
    
    # Build cache
    total_start = time.time()
    
    try:
        allocator = ResourceAllocator(
            log_path=LOG_PATH,
            permission_method='ordinor',
            n_trace_clusters=5,
            n_resource_clusters=10,
            cache_path=CACHE_FILE
        )
        
        total_elapsed = time.time() - total_start
        
        print("\n" + "="*70)
        print(f"✓ SUCCESS")
        print("="*70)
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"Cache saved to: {CACHE_FILE}")
        print(f"File size: {os.path.getsize(CACHE_FILE)/1024/1024:.1f} MB")
        print()
        print("You can now use this cache in your simulations:")
        print(f"  ResourceAllocator(log_path=..., cache_path='{CACHE_FILE}')")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print(f"[+] Cleaned up incomplete cache: {CACHE_FILE}")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print(f"[+] Cleaned up incomplete cache: {CACHE_FILE}")
        raise

if __name__ == "__main__":
    main()
