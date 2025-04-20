import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://www.americanyawp.com/"
OUTPUT_FILE = "american_yawp_paragraphs.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get_chapter_links(session):
    resp = session.get(BASE_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    chapters = []
    for toc_div in soup.find_all("div", id=lambda x: x and x.startswith("contents")):
        for a in toc_div.find_all("a", href=True):
            href = a["href"].strip()
            full_url = href if href.startswith("http") else urljoin(BASE_URL, href)
            title = a.get_text(strip=True)
            if full_url and title:
                chapters.append((title, full_url))
    seen = set()
    uniq = []
    for title, url in chapters:
        if url not in seen:
            uniq.append((title, url))
            seen.add(url)
    return uniq


def extract_paragraphs(html):
    soup = BeautifulSoup(html, "lxml")
    chapter_div = soup.find("div", class_="chapter") or soup.body
    paragraphs = []
    for p in chapter_div.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)
    return paragraphs


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    print("getting chapters...")
    chapters = get_chapter_links(session)
    print(f"find {len(chapters)} chapters")

    results = []
    for title, url in tqdm(chapters, desc="process"):
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"⚠️ cannot visit {url}: {e}")
            continue
        paras = extract_paragraphs(resp.text)
        for para in paras:
            results.append({
                "chapter": title,
                "paragraph": para
            })
        time.sleep(random.uniform(0.5, 1.5))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"finished! Store the result at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
