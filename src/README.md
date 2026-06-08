# README - Thư mục src

## 1) Giới thiệu
Thư mục src chứa mã nguồn cho pipeline RAG theo từng task (1 đến 10): thu thập dữ liệu, chuẩn hóa, truy xuất, rerank, fallback PageIndex và sinh câu trả lời có trích dẫn.

Xem yêu cầu chi tiết tại [README.md](../README.md).

## 2) Cách test
Chạy từ thư mục gốc project.

### Test toàn bộ
```bash
python -m pytest tests/test_individual.py -v
```

### Test từng task
```bash
python -m pytest tests/test_individual.py::TestTaskX -v
```

Ví dụ:
```bash
python -m pytest tests/test_individual.py::TestTask7 -v
python -m pytest tests/test_individual.py::TestTask10 -v
```

## 3) Cách chạy các file
Khuyến nghị chạy theo module:

```bash
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing
python -m src.task5_semantic_search
python -m src.task6_lexical_search
python -m src.task7_reranking
python -m src.task8_pageindex_vectorless
python -m src.task9_retrieval_pipeline
python -m src.task10_generation
```

## 4) Log chạy gần nhất
- python -m src.task10_generation: Exit Code 0
- python -m pytest tests/test_individual.py::TestTask10 -v: Exit Code 0
