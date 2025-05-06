import wikipediaapi
import json
import re
import math

from anyascii import anyascii
from collections import deque  

wiki = wikipediaapi.Wikipedia(
    language = "en",
    user_agent = "PythonScraper/1.0 (adriansuliga81@gmail.com)"
)

visited = set()

def get_stopwords(path: str) -> list:
    result = []
    file = open(path, "r")
    for line in file:
        result.append(line.strip())
    file.close()
    return result

def scrape(title: str, max_size: int, stopwords: list) -> dict:
    queue = deque([title])
    data, cnt = {}, 0

    while queue and len(visited) < max_size:
        current_title = queue.popleft()

        if current_title in visited: 
            print(f"Skipping {current_title} - already scraped")
            continue

        page = wiki.page(current_title)

        if not page.exists():
            print(f"Skipping {current_title} - page not found")
            continue

        visited.add(current_title)
        cnt += 1

        print(f"[{(cnt * 100) / max_size}%] " 
              f"Scraping {page.title} ({title_to_url(page.title)})")

        bow = {}

        for word in page.text.split():
            parsed_word = anyascii(word)
            parsed_word = re.sub(
                r'[!@#$%^*|;?><:`.,()\[\]{}\/\\\"\']',
                '',
                parsed_word
            ).lower()

            if parsed_word in stopwords or parsed_word == "":
                continue

            bow[parsed_word] = 1 + bow.get(parsed_word, 0)

        data[title_to_url(page.title)] = bow

        for new_title in page.links.keys():
            if new_title not in visited:
                queue.append(new_title)

    return data

def title_to_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

def read_dict_from_json(path: str) -> dict:
    file = open(path, "r")
    result = json.load(file)
    file.close()
    return result

def save_dict_to_json(data: dict, path: str) -> None:
    file = open(path, "w")
    json.dump(data, file, indent = 4)
    file.close()

def create_bows(urls: dict) -> None:
    url_indexes = {}
    url_index = 0
    global_bow = {}

    for url in list(urls.keys()):
        url_indexes[url] = url_index
        url_index += 1
        global_bow.update(urls[url])

    bow_index = 0

    for word in list(global_bow.keys()):
        global_bow[word] = bow_index
        bow_index += 1

    save_dict_to_json(urls, "database/urls.json")
    save_dict_to_json(global_bow, "database/global_bow.json")

def index_urls() -> None:
    urls = read_dict_from_json("database/urls.json")
    url_indexes = {}
    indexed_urls = {}
    current_index = 0

    for url, _ in urls.items():
        url_indexes[url] = current_index
        indexed_urls[current_index] = url
        current_index += 1

    save_dict_to_json(url_indexes, "database/url_indexes.json")
    save_dict_to_json(indexed_urls, "database/indexed_urls.json")

def calculate_idfs() -> dict:
    occurances = {}
    urls = read_dict_from_json("database/urls.json")

    for _, url_dict in urls.items():
        for word in list(url_dict.keys()):
            occurances[word] = 1 + occurances.get(word, 0)
    
    n = len(urls)
    idf = {}

    for key in occurances:
        idf[key] = math.log10(n / occurances[key])

    save_dict_to_json(idf, "database/idfs.json")

def main() -> None:
    data = scrape("Brandon_Sanderson", 100, get_stopwords("stopwords.txt"))

    create_bows(data)
    index_urls()
    calculate_idfs()

if __name__ == "__main__":
    main()
