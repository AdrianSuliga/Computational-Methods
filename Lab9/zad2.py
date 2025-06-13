from PIL import Image, ImageOps
from copy import deepcopy
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import numpy as np
import cv2

ALPHABET = [str(chr(c)) for c in range(ord('a'), ord('z') + 1)] + \
            [str(chr(c)) for c in range(ord('0'), ord('9') + 1)] + \
            [".", ",", "!", "?", "(", ")", " "]
CHAR_IMG_DICT = {}
SPECIAL_CHARS_TO_FILE = {".": "A", ",": "B", "!": "C", "?": "D", "(": "E", ")": "F", " ": "G"}
FILE_TO_SPECIAL_CHARS = {"A": ".", "B": ",", "C": "!", "D": "?", "E": "(", "F": ")", "G": " "}

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

# IMAGE PREPARATION (OK)
def show_image(image, title):
    plt.imshow(image, cmap = "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def negate(image):
    w, h = image.size
    pix = image.load()

    for y in range(h):
        for x in range(w):
            pix[x, y] = 255 - pix[x, y]
    
    return image

def negate_save_image(path: str, font: str):
    font = font.lower()
    img = Image.open(path).convert("L")
    img = negate(img)
    
    inverted_path = path.replace(font, "inverted_" + font)

    img.save(inverted_path)

    show_image(img, f"Inverted image with path {path}")

    return Image.open(inverted_path).convert("L")

def show_alphabet(alphabet: str, folder: str) -> dict:
    path = folder + alphabet.lower() + "/"
    letter_paths = []

    for char in ALPHABET:
        if char in SPECIAL_CHARS_TO_FILE.keys():
            letter_paths.append((path + SPECIAL_CHARS_TO_FILE[char] + ".png", char))
        else:
            letter_paths.append((path + char + ".png", char))

    rows, cols = 5, 10

    fig, axes = plt.subplots(rows, cols)
    fig.suptitle(f"Alphabet Set: {alphabet[:-1].capitalize()}", fontsize = 16)
    axes = axes.flatten()

    mark = 0

    for i, (image_path, char) in enumerate(letter_paths):
        img = negate(Image.open(image_path).convert("L"))
        CHAR_IMG_DICT[char] = img
        axes[i].imshow(img, cmap = "gray")
        axes[i].axis("off")
        mark = i

    for j in range(mark + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    cos = np.abs(np.cos(angle))
    sin = np.abs(np.sin(angle))

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M = cv2.getRotationMatrix2D((center_x, center_y), angle * 180 / np.pi, 1.0)
    M[0, 2] += (new_w / 2) - center_x
    M[1, 2] += (new_h / 2) - center_y
    
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags = cv2.INTER_NEAREST,
        borderMode = cv2.BORDER_REPLICATE
    )

    return rotated

def whiten(image_array):
    mask = np.any(image_array != BLACK, axis=-1)
    image_array[mask] = WHITE
    return image_array

def cut_image(image):
    image_array = np.array(image)

    non_white_cols = np.any(np.any(image_array != BLACK, axis = -1), axis = 1)
    non_white_rows = np.any(np.any(image_array != BLACK, axis = -1), axis = 0)

    minimal_row, maximal_row = float('inf'), float('-inf')
    minimal_col, maximal_col = float('inf'), float('-inf')

    for i in range(len(non_white_rows)):
        val = non_white_rows[i]
        if val:
            minimal_row = min(minimal_row, i)
            maximal_row = max(maximal_row, i + 1)

    for i in range(len(non_white_cols)):
        val = non_white_cols[i]
        if val:
            minimal_col = min(minimal_col, i)
            maximal_col = max(maximal_col, i + 1)
    
    x_range = maximal_row - minimal_row
    y_range = maximal_col - minimal_col
    
    new_image_array = np.zeros((y_range, x_range, 3))
    
    for x in range(x_range):
        for y in range(y_range):
            new_image_array[y, x] = image_array[minimal_col + y, minimal_row + x]
    
    return whiten(new_image_array.astype(np.uint8))

def straighten_image(image: Image.Image):
    image = np.array(image.convert("RGB"))
    edges = cv2.Canny(image, 50, 150, apertureSize = 3)
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    angles = []
    if lines is not None:
        for _, angle in lines[:, 0]:
            angles.append(angle - np.pi / 2)
    
    mean_angle = np.mean(angles) if angles else 0
    
    new_image = cut_image(rotate_image(image, mean_angle))
    straightened_image = cv2.fastNlMeansDenoisingColored(new_image, None, 10, 10, 10, 20)

    straightened_image = Image.fromarray(straightened_image).convert("L")
    straightened_image = ImageOps.expand(straightened_image, border = 20, fill = 0)
    show_image(straightened_image, "Wyprostowany obraz")

    return straightened_image

def combine_letters(path: str) -> str:
    images_paths = []

    for char in ALPHABET:
        if char in SPECIAL_CHARS_TO_FILE.keys():
            images_paths.append(path + SPECIAL_CHARS_TO_FILE[char] + ".png")
        else: 
            images_paths.append(path + str(char) + ".png")

    images = [negate(Image.open(path).convert("L")) for path in images_paths] 
    max_height = max(img.height for img in images)

    padded_images = []
    for img in images:
        if img.height < max_height:
            new_img = Image.new("L", (img.width, max_height), color = 0)
            new_img.paste(img, (0, 0))
            padded_images.append(new_img)
        else:
            padded_images.append(img)
    
    total_width = sum(img.width for img in padded_images)
    combined_img = Image.new("L", (total_width, max_height), color = 0)

    x_offset = 0
    for img in padded_images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    output_path = IMG_FOLDER + path.split("/")[-2] + "_alphabet.png"
    combined_img.save(output_path)

    show_image(combined_img, "Połączony alfabet")

    return output_path

def add_alphabet(image: str, alphabet_path: str) -> str:
    image_to_paste_path = combine_letters(alphabet_path)
    alphabet_img = Image.open(image_to_paste_path).convert("L")

    new_height = image.height + alphabet_img.height

    combined_img = Image.new("L", (image.width, new_height), color = 0)
    combined_img.paste(image, (0, 0))
    combined_img.paste(alphabet_img, ((image.width - alphabet_img.width) // 2, image.height))
    combined_img.save(IMG_FOLDER + "alphabet_added_text.png")
    
    show_image(combined_img, "Obraz z dodanym alfabetem")

    return combined_img

# PATTERN SEARCHING
def find_corelation(image, pattern):
    h, w = image.shape
    correlation = np.fft.ifft2(
        np.fft.fft2(image) * np.fft.fft2(np.rot90(np.rot90(pattern)), s = (h, w))
    ).real
    correlation /= np.abs(np.max(correlation))
    return correlation

def is_maximum(correlation, x, y, previously_found):
    h, w = correlation.shape
    val = correlation[x, y]
    local_max = (x, y)
    stack = [local_max]
    
    while stack:
        # Pomijamy już znalezione
        current_x, current_y = stack.pop()
        if (current_x, current_y) in previously_found: continue
        previously_found.add((current_x, current_y))

        neighbors = [
            (i, j) for i in range(max(0, current_x - 1), min(current_x + 2, h))
                   for j in range(max(0, current_y - 1), min(current_y + 2, w))
                   if (i, j) not in previously_found
        ]

        for nx, ny in neighbors:
            # Jeśli nie jest maksimum, to poddajemy się
            if correlation[nx, ny] > val: return False, None
            # Jeśli inny punkt ma taką samą korelację to sprawdzamy go
            # po zakończeniu pętli zostaniemy z ostatnim znalezionym punktem
            # o odpowiedniej korelacji
            if correlation[nx, ny] == val:
                stack.append((nx, ny))
                if (ny, nx) > (local_max[1], local_max[0]):
                    local_max = (nx, ny)
                    
    return True, local_max

def handle_letter(correlation, prob, pattern, image):
    pattern_h, pattern_w = pattern.shape
    height = image.shape[0]
    max_points = np.argwhere(correlation > prob)
    
    sorted_indices = np.argsort(correlation[max_points[:, 0], max_points[:, 1]])[::-1]
    max_points = max_points[sorted_indices]
    
    new_points = set()
    
    for x, y in max_points:
        if (x, y) in new_points: continue
        is_max, point = is_maximum(correlation, x, y, set())
        if is_max: new_points.add(point)
    
    max_points = list(new_points)
    max_points.sort(key = lambda x: correlation[x], reverse = True)
    
    appearances = []
    
    # Malujemy na czarno fragmenty uznane za rozważaną literę
    for x, y in max_points:
        point = (y - pattern_w, x - pattern_h)
        if x <= height: appearances.append(point)
        image[max(0, point[1]) : x + 1, max(0, point[0]) : y + 1] = 0
    
    return appearances, image

def count_blacks(image):
    w, h = image.size
    pix = image.load()

    cnt = 0

    for y in range(h):
        for x in range(w):
            if pix[x, y] == 255: cnt += 1
    
    return cnt

def sort_characters(characters):
    ordered_characters = list(characters)
    ordered_characters.sort(key = lambda x: count_blacks(CHAR_IMG_DICT[x]), reverse=True)
    return ordered_characters

def scan_image(image):
    ordered_characters = sort_characters(CHAR_IMG_DICT)
    ordered_characters.pop()
    
    character_data = {}
    
    for character in ordered_characters :
        pattern = np.array(CHAR_IMG_DICT[str(character)])
        correlation = find_corelation(image, pattern)
        character_data[character], image = handle_letter(correlation, 0.84, pattern, image)

    return character_data

# IMAGE UNDERSTANDING
def create_point_char_dict(character_data):
    point_char = {}

    for key, values in character_data.items():
        for point in values:
            point_char[point] = key

    return point_char

def find_first_point(points, line_height):
    # Znajdź najbardziej górny punkt
    upmost_point = min(points, key = lambda x: (x[1], x[0]))
    
    def find(point):
        nonlocal upmost_point, line_height
        if point[1] < upmost_point[1] + line_height: return point[0]
        return float('inf')

    # Znajdź punkt najbardziej na lewo
    leftmost_point = min(points, key = lambda x : find(x))
            
    return leftmost_point

def order_points(points, height):
    points = deepcopy(points)
    
    lines = []
    while points:
        # Bierzemy lewy-górny znak
        first_point = find_first_point(points, height)
        line = [first_point]

        # Dodajemy wszystkie wystąpienia w ramach aktualnej linii
        for point in points:
            if point == first_point: continue
            if np.abs(point[1] - first_point[1]) < height:
                line.append(point)

        # Sortujemy po x-ach
        line.sort(key = lambda x : x[0])
        
        # Usuwamy aktualną (najwyższą) linię
        for point in line:
            points.remove(point)
        
        lines.append(line)
    
    return lines

def arrange_letters(data, space_multiplicator):
    # Obliczamy wysokość linii i szerokość spacji
    space_width, space_height = CHAR_IMG_DICT[" "].size
    space_width = int(round(space_width * space_multiplicator))
    
    # Grupujemy punkty w liniach
    points_to_char = create_point_char_dict(data)
    points = list(points_to_char)
    lines = order_points(points, space_height)
    text = ""
    
    if (len(lines) < 1): return text
    
    line_height = lines[1][0][1] - lines[0][0][1]
    
    # Budujemy końcowy napis
    for i, line in enumerate(lines):
        prev = line[0]
        x, y = prev
        letter = points_to_char[prev]
        text += letter
        end_x = CHAR_IMG_DICT[letter].size[0] + x

        for point in line:
            if point == prev: continue

            x = point[0]
            space_count = (x - end_x) // space_width
            
            for _ in range(space_count):
                text += " "
            
            letter = points_to_char[point]
            text += letter
            end_x = CHAR_IMG_DICT[letter].size[0] + x
            prev = point

        if i != len(lines) - 1:
            next_y = lines[i + 1][0][1]
            for _ in range(round(((next_y - y) / line_height))): text += "\n"
        else: break
    
    return text

def similarity(s1: str, s2: str):
    lev_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (lev_dist / max_len)

IMG_FOLDER = "images_ocr/"
ALPHABET_FOLDER = IMG_FOLDER + "alphabets/"
chosen_font = "TimesNewRoman"
text = "timesnewroman_text_3.txt"
img_path = "timesnewroman_text_3.png"

inverted_image = negate_save_image(IMG_FOLDER + img_path, chosen_font)
show_alphabet(chosen_font, ALPHABET_FOLDER)
straightened_image = straighten_image(inverted_image)
image_with_alphabet = add_alphabet(straightened_image, ALPHABET_FOLDER + chosen_font.lower() + "/")
result = scan_image(np.array(image_with_alphabet))
read_text = arrange_letters(result, 0.4)

print(read_text)
print(
    "Similarity level:",
    str(round(similarity(read_text, open(IMG_FOLDER + text).read()) * 100, 2)) + "%"
)
