"""
Run this manually whenever you want to refresh vector DB
Usage:
    python rebuild_vectors.py
"""

from knowledge_base import rebuild_vector_db

print("\n🔧 MANUAL VECTOR DB REBUILD TOOL")
print("=================================")

confirm = input("Rebuild vector database now? (yes/no): ")

if confirm.lower() == "yes":
    rebuild_vector_db()
else:
    print("❌ Cancelled — no changes made.")
