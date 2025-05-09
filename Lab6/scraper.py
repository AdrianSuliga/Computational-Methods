import numpy as np

import wikipediaapi, json, time, random, requests, re, math

from anyascii import anyascii
from collections import deque
from scipy import sparse
from scipy.sparse import linalg
from sklearn.decomposition import TruncatedSVD

wiki = wikipediaapi.Wikipedia(
    language = "en",
    user_agent = "PythonScraper/1.0 (adriansuliga81@gmail.com)"
)

def get_stopwords(path: str) -> list:
    result = []
    file = open(path, "r")
    for line in file:
        result.append(line.strip())
    file.close()
    return result

def title_to_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

def scrape(titles: list, max_size: int, stopwords: list) -> dict:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries = 3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    session.request = lambda *args, **kwargs: requests.Session.request(
        session,
        *args,
        timeout = 30,
        **kwargs
    )

    queue = deque(titles)
    data, cnt = {}, 0
    visited = set()
    broken = set()

    while queue and len(visited) < max_size:
        current_title = queue.popleft()

        if current_title in visited or current_title in broken: 
            print(f"Skipping {current_title} - already visited")
            continue

        try:
            page = wiki.page(current_title)

            if not page.exists():
                broken.add(current_title)
                print(f"Skipping {current_title} - page not found")
                continue

            visited.add(current_title)
            cnt += 1

            print(f"[{round((cnt * 100) / max_size, 3)}%] " 
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
        except Exception as e:
            print(f"Error while scraping {current_title}: {e}")
            broken.add(current_title)
        
        time.sleep(random.uniform(1, 3))

    return data

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

def create_tbd_matrix() -> sparse.csr_matrix:
    print("Creating TBD matrix")
    urls = read_dict_from_json("database/urls.json")
    global_bow = read_dict_from_json("database/global_bow.json")
    idfs = read_dict_from_json("database/idfs.json")
    url_indexes = read_dict_from_json("database/url_indexes.json")

    tbd = sparse.lil_matrix(
        (len(global_bow), len(urls)),
        dtype = np.float32
    )

    print("Filling TBD matrix")
    for url, url_dict in urls.items():
        for word, value in url_dict.items():
            tbd[global_bow[word], url_indexes[url]] = value * idfs[word]

    print("Normalising TBD")
    tbd_csc = tbd.tocsc()
    column_norms = linalg.norm(tbd_csc, axis = 0)
    normalised_tbd = tbd_csc / column_norms

    print("TBD done")
    return normalised_tbd

def initialize_svd(search_matrix, k):
    print("SVD calculation started.")
    svd = TruncatedSVD(n_components = k)
    svd.fit(search_matrix)

    us_matrix = svd.transform(search_matrix)
    v_t_matrix = np.array(svd.components_)
    print("SVD calculation finished.")

    with open("database/us.npz", 'wb') as file:
        np.save(file, us_matrix)

    with open("database/v_t.npz", 'wb') as file:
        np.save(file, v_t_matrix)

    print("SVD matrices saved succesfully.")

def input_to_vector(input: str) -> sparse.csr_matrix:
    print("Parsing input")
    input = anyascii(input.lower())
    parsed_words = []
    for word in input.split():
        parsed_words.append(
            re.sub(
                    r'[!@#$%^*|;?><:`.,()\[\]{}\/\\\"\']',
                    '',
                    word
                ).lower()
        )

    stopwords = get_stopwords("database/stopwords.txt")
    filtered_words = []

    for word in parsed_words:
        if word not in stopwords: 
            filtered_words.append(word)

    q = {}

    for word in filtered_words:
        q[word] = 1 + q.get(word, 0)

    global_bow = read_dict_from_json("database/global_bow.json")
    idfs = read_dict_from_json("database/idfs.json")

    vector = sparse.lil_matrix((len(global_bow), 1), dtype = np.float32)

    for word, value in q.items():
        if word not in global_bow: continue
        vector[global_bow[word], 0] = value * idfs[word]

    print("Input parsed")
    return sparse.csr_matrix(vector.tocsr())

def search(tbd_matrix: sparse.csr_matrix, input: str, k: int = 10, use_svd: bool = False) -> list:
    print(f"Searching for \"{input}\"")
    indexed_urls = read_dict_from_json("database/indexed_urls.json")
    if input == "": return []

    query = input_to_vector(input)
    query_result = {}
    cosines = []

    if use_svd:
        u_times_s = np.load("database/us.npz")
        v_transposed = np.load("database/v_t.npz")

        query /= linalg.norm(query)

        cosines = ((np.transpose(query.toarray())) @ u_times_s) @ v_transposed
    else:
        query /= linalg.norm(query)
        cosines = query.transpose() @ tbd_matrix

    for j in range(cosines.shape[1]):
        query_result[indexed_urls[str(j)]] = cosines[0, j]

    sorted_result = sorted(
        query_result.items(),
        key = lambda x: x[1],
        reverse = True
    )

    return sorted_result[:k]

def rescrape_wikipedia():
    data = scrape(
        [
            "Karl_Marx",
            "Iron_Man",
            "The_Walt_Disney_Company",
            "Brandon_Sanderson",
            "Lionel_Messi",
            "Horse",
            "Rise_Against"
        ],
        20000,
        get_stopwords("database/stopwords.txt")
    )

    create_bows(data)
    index_urls()
    calculate_idfs()

