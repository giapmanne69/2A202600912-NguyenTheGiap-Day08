import os
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def load_corpus_local() -> list[dict]:
    corpus_data = []
    if not STANDARDIZED_DIR.exists():
        return corpus_data

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file.parts) else "news"
            
            # Tách nhỏ nội dung thành các đoạn văn bản tương tự cấu trúc chunk của Task 4
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, p in enumerate(paragraphs):
                corpus_data.append({
                    "content": p,
                    "metadata": {
                        "source": md_file.name,
                        "type": doc_type,
                        "chunk_index": i
                    }
                })
        except Exception:
            continue
    return corpus_data


CORPUS: list[dict] = load_corpus_local()


def build_bm25_index(corpus: list[dict]):
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# Khởi tạo mô hình BM25 từ corpus dữ liệu local
bm25_model = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    if not CORPUS:
        return []

    tokenized_query = query.lower().split()
    scores = bm25_model.get_scores(tokenized_query)

    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": {
                    "source": CORPUS[idx]["metadata"]["source"],
                    "doc_type": CORPUS[idx]["metadata"]["type"],
                    "chunk_index": CORPUS[idx]["metadata"]["chunk_index"]
                }
            })
    return results


if __name__ == "__main__":
    print(f"✓ Corpus local: {len(CORPUS)} chunks.")
    test_query = "Điều 248 tàng trữ trái phép chất ma tuý"
    results = lexical_search(test_query, top_k=5)
    
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")