from flask import Flask, send_from_directory, request, render_template
import time, threading, re

from scraper import create_tbd_matrix, search, initialize_svd

print("Creating server app")
app = Flask(__name__, static_folder='static')

tbd = create_tbd_matrix()
initialize_svd(tbd, 100)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/search', methods=['GET'])
def search_callback():
    global tbd

    query = request.args.get('q', '')
    k = request.args.get('output_size', 10)
    svd = request.args.get('use_svd')

    try:
        k = int(k)
    except ValueError:
        k = 10

    if bool(svd):
        print("Using SVD for better search")

    start_time = time.time()
    result = search(tbd, query, k=k, use_svd=bool(svd))
    end_time = time.time()
    response_time = f"{end_time - start_time:.2f} seconds"

    links = extract_links(result) if result else []
    print(links)

    return render_template('search.html', links=links, input_string=query, response_time=response_time)

def extract_links(result_list):
    return [{
        'title': re.sub('_', ' ', url.split("/")[-1]),
        'url': url,
        'value': val
        } for url, val in result_list]

def run_app():
    app.run(host='0.0.0.0', port=4000, debug=False, use_reloader=False)

thread = threading.Thread(target = run_app)
thread.start()
