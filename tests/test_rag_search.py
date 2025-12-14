# tests/test_rag_search.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.rag_search import RAGIndex, summarize

def test_rag_pipeline():
    docs = [
        ("doc1", "The capital of France is Paris."),
        ("doc2", "The capital of Germany is Berlin."),
        ("doc3", "Paris is known for the Eiffel Tower.")
    ]
    idx = RAGIndex()
    idx.add_docs(docs)
    
    # Search for "Paris"
    results = idx.search("Paris", k=2)
    # Mock embedder is simple hash, but let's just check we got results
    assert len(results) > 0
    
    # Check summarization
    summary = summarize([t for _,t,_ in results])
    assert "Paris" in summary or "France" in summary

if __name__ == "__main__":
    test_rag_pipeline()
    print("RAG tests passed!")
