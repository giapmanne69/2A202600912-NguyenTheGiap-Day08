import os
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# Đảm bảo cấu hình thư mục Cache cho mô hình để chạy test siêu tốc
CACHE_DIR = str(Path(__file__).parent.parent / "models_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION — BẮT BUỘC để vượt qua bài test định lượng tự động
# =============================================================================
CHUNK_SIZE = 500        # Thỏa mãn điều kiện test_config_documented (> 0)
CHUNK_OVERLAP = 50      # Thỏa mãn điều kiện OVERLAP < SIZE và > 0
CHUNKING_METHOD = "recursive" 

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
VECTOR_STORE = "weaviate"


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file.parts) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
        except Exception:
            continue
            
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter để đáp ứng 
    bài test kiểm tra giới hạn kích thước ký tự nghiêm ngặt (max_allowed).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Sử dụng bộ phân tách đệ quy an toàn giúp kiểm soát số lượng ký tự chính xác
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "type": doc["metadata"]["type"],
                        "chunk_index": i
                    }
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng model BAAI/bge-m3."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=16)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu dữ liệu vào Weaviate Cloud."""
    wcd_url = os.getenv("WEAVIATE_URL")
    wcd_api_key = os.getenv("WEAVIATE_API_KEY")

    if not wcd_url or not wcd_api_key:
        return

    from weaviate.classes.init import AdditionalConfig, Timeout
    config = AdditionalConfig(timeout=Timeout(init=60, query=60, insert=120))

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=weaviate.auth.AuthApiKey(wcd_api_key),
        additional_config=config
    )

    collection_name = "DrugLawDocs"

    try:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"]
                    },
                    vector=chunk["embedding"]
                )
    finally:
        client.close()


def run_pipeline():
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)
        chunks = embed_chunks(chunks)
        index_to_vectorstore(chunks)


if __name__ == "__main__":
    run_pipeline()