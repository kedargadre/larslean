"""Quick smoke test for the AI Advisor pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 1. Test RAG engine
print("=" * 60)
print("1. Testing Hybrid RAG Engine")
print("=" * 60)

from insights.rag_engine import HybridRAGEngine

engine = HybridRAGEngine()
print(f"   Chunks loaded: {len(engine.chunks)}")
print(f"   TF-IDF matrix: {engine.tfidf_matrix.shape if engine.tfidf_matrix is not None else 'None'}")
print(f"   Gemini embeddings: {'Yes' if engine._gemini_available else 'No (TF-IDF only)'}")
if engine.chunk_embeddings is not None:
    print(f"   Embedding shape: {engine.chunk_embeddings.shape}")

# 2. Test retrieval
print("\n" + "=" * 60)
print("2. Testing Retrieval: 'affordable rent Dublin'")
print("=" * 60)

results = engine.retrieve("affordable rent Dublin", top_k=3)
for i, r in enumerate(results, 1):
    print(f"\n   [{i}] {r['source_title']} (score: {r['score']:.4f}, method: {r['method']})")
    print(f"       {r['content'][:120]}...")

# 3. Test retrieval: policy question
print("\n" + "=" * 60)
print("3. Testing Retrieval: 'what grants are available for first time buyers'")
print("=" * 60)

results2 = engine.retrieve("what grants are available for first time buyers", top_k=3)
for i, r in enumerate(results2, 1):
    print(f"\n   [{i}] {r['source_title']} (score: {r['score']:.4f}, method: {r['method']})")
    print(f"       {r['content'][:120]}...")

# 4. Test retrieval: energy/BER
print("\n" + "=" * 60)
print("4. Testing Retrieval: 'energy efficiency rural homes'")
print("=" * 60)

results3 = engine.retrieve("energy efficiency rural homes", top_k=3)
for i, r in enumerate(results3, 1):
    print(f"\n   [{i}] {r['source_title']} (score: {r['score']:.4f}, method: {r['method']})")
    print(f"       {r['content'][:120]}...")

# 5. Test context builder
print("\n" + "=" * 60)
print("5. Testing Context Builder")
print("=" * 60)

from insights.context import build_page_context, build_area_context
import pandas as pd

ctx = build_page_context(
    active_tab="forecast",
    selected_county="Cork",
    selected_metric="risk_score",
)
print(f"   Tab: {ctx['active_tab']}")
print(f"   Description: {ctx['natural_description']}")

# 6. Test Gemini LLM call
print("\n" + "=" * 60)
print("6. Testing Gemini LLM (gemini-2.5-flash)")
print("=" * 60)

try:
    from google import genai
    from config import GEMINI_API_KEY

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Learslan AI is online' in one sentence.",
    )
    print(f"   Response: {response.text.strip()}")
    print("   Gemini connection successful!")
except Exception as e:
    print(f"   Gemini failed: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
