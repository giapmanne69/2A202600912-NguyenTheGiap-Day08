import os
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

# Đảm bảo cấu hình thư mục Cache cho mô hình để load siêu tốc từ ổ cứng
CACHE_DIR = str(Path(__file__).parent.parent / "models_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

load_dotenv()

# Khởi tạo mô hình Embedding BAAI/bge-m3 (giống hệt Task 4) ở phạm vi toàn cục để tránh bị re-load mỗi lần gọi hàm
EMBEDDING_MODEL = "BAAI/bge-m3"
print(f"-> Đang nạp mô hình Semantic Search {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity trên Weaviate Cloud.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    wcd_url = os.getenv("WEAVIATE_URL")
    wcd_api_key = os.getenv("WEAVIATE_API_KEY")

    if not wcd_url or not wcd_api_key:
        raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env configuration!")

    # Bước 1: Trích xuất vector (Embed) câu truy vấn của người dùng
    query_embedding = model.encode(query).tolist()

    # Kết nối tới Weaviate Cloud với cấu hình timeout an toàn
    from weaviate.classes.init import AdditionalConfig, Timeout
    config = AdditionalConfig(timeout=Timeout(init=20, query=20))
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=weaviate.auth.AuthApiKey(wcd_api_key),
        additional_config=config
    )

    collection_name = "DrugLawDocs"
    output_results = []

    try:
        # Kiểm tra xem bộ chỉ mục dữ liệu đã được tạo từ Task 4 chưa
        if not client.collections.exists(collection_name):
            print(f"⚠ Không tìm thấy collection '{collection_name}'. Hãy chạy Task 4 trước!")
            return []

        collection = client.collections.get(collection_name)

        # Bước 2: Query dữ liệu sử dụng toán tử không gian vector (Vector Search)
        # Weaviate Cloud tính toán khoảng cách mặc định là Cosine Distance (Càng gần 0 càng giống nhau)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)  # Bắt buộc lấy thuộc tính distance để quy đổi score
        )

        # Bước 3: Đóng gói dữ liệu đầu ra và chuyển đổi Distance thành Similarity Score
        for obj in results.objects:
            props = obj.properties
            
            # Quy đổi từ khoảng cách (Distance) sang độ tương đồng (Similarity Score)
            # Với cấu hình mặc định của Weaviate: Score = 1 - Cosine_Distance
            score = 1.0 - float(obj.metadata.distance) if obj.metadata.distance is not None else 0.0

            output_results.append({
                "content": props.get("content", ""),
                "score": score,
                "metadata": {
                    "source": props.get("source", "Unknown"),
                    "doc_type": props.get("doc_type", "Unknown"),
                    "chunk_index": int(props.get("chunk_index", 0)) if props.get("chunk_index") is not None else 0
                }
            })

        # Sắp xếp lại kết quả theo thứ tự điểm Score giảm dần (Sorted Descending)
        output_results = sorted(output_results, key=lambda x: x["score"], reverse=True)

    except Exception as e:
        print(f"✗ Lỗi trong quá trình thực hiện Semantic Search: {e}")
    finally:
        client.close()

    return output_results


if __name__ == "__main__":
    print("\n=== ĐANG CHẠY THỬ MODULE SEMANTIC SEARCH ===")
    test_query = "hình phạt cho tội tàng trữ ma tuý"
    results = semantic_search(test_query, top_k=5)
    
    print(f"\nKết quả tìm kiếm cho query: '{test_query}':\n")
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] Điểm số tương đồng: {r['score']:.3f}")
        print(f"    Nguồn: {r['metadata']['source']} | Phân loại: {r['metadata']['doc_type']}")
        print(f"    Nội dung: {r['content'][:150]}...\n")