import os
import requests
from typing import Optional

# Tải cấu hình biến môi trường cục bộ (.env)
from dotenv import load_dotenv
load_dotenv()


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina AI Cross-Encoder API v2 (Multilingual).
    """
    if not candidates:
        return []

    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("Missing JINA_API_KEY in .env configuration!")

    # Chuẩn bị payload danh sách chuỗi văn bản gửi lên API xử lý
    documents = [c["content"] for c in candidates]

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {jina_api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": top_k
            },
            timeout=20
        )
        response.raise_for_status()
        reranked_results = response.json()["results"]

        # Trích xuất vị trí index cũ và cập nhật lại điểm số Relevance Score mới từ Cross-Encoder
        output_candidates = []
        for r in reranked_results:
            orig_idx = r["index"]
            updated_item = candidates[orig_idx].copy()
            updated_item["score"] = float(r["relevance_score"])
            output_candidates.append(updated_item)
            
        return output_candidates

    except Exception as e:
        print(f"✗ Lỗi kết nối Jina Reranker API: {e}. Fallback sang trả về top_k gốc.")
        # Nếu API lỗi, fallback sắp xếp danh sách cũ theo điểm số Retrieval và trả về top_k
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]


def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Hàm bổ trợ tính toán Cosine Similarity giữa 2 vector."""
    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — cân bằng giữa mức độ liên quan và tính đa dạng thông tin.
    """
    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    # Vòng lặp chọn ra top_k phần tử tối ưu
    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Đo độ tương đồng ngữ nghĩa của Candidate với câu hỏi gốc
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Tìm độ trùng lặp lớn nhất của Candidate này đối với các phần tử đã được chọn trước đó
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # Công thức tính điểm MMR chuẩn hóa
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — Thuật toán hợp nhất thứ hạng không cần chuẩn hóa điểm số.
    """
    rrf_scores = {}  # Lưu trữ cặp khóa-giá trị: content -> RRF score
    content_map = {} # Phục vụ map ngược lại metadata gốc từ chuỗi content

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            # Công thức RRF tích lũy vị trí xếp hạng trên toàn bộ các bảng kết quả tìm kiếm độc lập
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sắp xếp danh sách hợp nhất theo điểm số RRF tích lũy giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface phục vụ cho toàn bộ Pipeline.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Hàm MMR thuần túy yêu cầu truyền query_embedding trực tiếp
        raise NotImplementedError("Hãy gọi trực tiếp hàm 'rerank_mmr' và truyền kèm tham số vector query_embedding.")
    elif method == "rrf":
        # Hàm RRF yêu cầu truyền mảng tập hợp các bảng xếp hạng độc lập
        raise NotImplementedError("Hãy gọi trực tiếp hàm 'rerank_rrf' và truyền kèm tham số mảng kết quả ranked_lists.")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Điền API Key thử nghiệm trực tiếp nếu chạy riêng lẻ file script này
    if not os.getenv("JINA_API_KEY"):
        os.environ["JINA_API_KEY"] = "jina_YOUR_API_KEY_HERE"

    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    
    print("=== ĐANG CHẠY THỬ MODULE RERANKING (JINA API) ===")
    try:
        results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content']}")
    except Exception as e:
        print(f"Bỏ qua demo lỗi (Do chưa điền API Key thật): {e}")