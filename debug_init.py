
import sys
import os
import traceback

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from backend.rag_search import initialize_system

print("Debugging Initialization...")
try:
    initialize_system()
    print("Success!")
except Exception:
    traceback.print_exc()
