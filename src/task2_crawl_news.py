import json
import time
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://tuoitre.vn/ca-si-long-nhat-va-son-ngoc-minh-mua-ma-tuy-tu-dau-20260522101239371.htm",
    "https://tienphong.vn/bi-hai-chuyen-nghe-si-test-ma-tuy-post1847129.tpo",
    "https://tuoitre.vn/vu-miu-le-long-nhat-son-ngoc-minh-nghe-si-phai-giu-hinh-anh-chin-chu-tren-san-khau-lan-ngoai-doi-2026052112085492.htm",
    "https://thanhnien.vn/nghe-si-tu-nguyen-xet-nghiem-ma-tuy-showbiz-dang-bat-an-den-muc-nao-185260526105918638.htm",
]


def crawl_article(url: str) -> dict:
    """Crawl một bài báo bằng requests + BeautifulSoup và trả về dict."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Khởi tạo giá trị mặc định phòng trường hợp lỗi
    title = "Unknown"
    content_markdown = ""

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = "utf-8"  # Đảm bảo không bị lỗi font tiếng Việt

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # 1. Lấy Tiêu đề bài báo (bóc tách từ thẻ <title> hoặc thẻ h1 bất kỳ)
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()

            # 2. Bóc tách nội dung chính của bài báo
            # Gom toàn bộ các đoạn văn bản (thẻ <p>) lại thành nội dung text giả lập markdown
            paragraphs = soup.find_all("p")
            text_lines = []
            for p in paragraphs:
                p_text = p.get_text().strip()
                # Lọc bỏ các đoạn chữ quá ngắn hoặc rác (như menu, copyright...)
                if len(p_text) > 20:
                    text_lines.append(p_text)

            content_markdown = "\n\n".join(text_lines)
        else:
            print(f"  ✗ Lỗi HTTP {response.status_code} khi tải {url}")

    except Exception as e:
        print(f"  ✗ Lỗi kết nối khi crawl {url}: {e}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS (Đồng bộ)."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath}")

        # Nghỉ 1 giây giữa các lần crawl để tránh bị chặn IP (Rate limit)
        time.sleep(1)


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        crawl_all()