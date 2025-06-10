from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_image(image, title):
    plt.imshow(image, cmap = "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def invert_image(path: str, font: str):
    img = Image.open(path).convert("L")
    w, h = img.size
    pix = img.load()

    for y in range(h):
        for x in range(w):
            pix[x, y] = 255 - pix[x, y]
    
    font = font[:-1]
    inverted_path = path.replace(font, "inverted_" + font)
    
    img.save(inverted_path)

    show_image(img, f"Inverted image with path {path}")

    return inverted_path

def show_alphabet(alphabet: str, folder: str):
    path = folder + alphabet
    letter_paths = []

    for i in range(97, 123):
        letter_paths.append(path + str(chr(i)) + ".png")
    
    for i in range(48, 58):
        letter_paths.append(path + str(chr(i)) + ".png")

    letter_paths.append(path + "?.png")
    letter_paths.append(path + "!.png")
    letter_paths.append(path + "..png")
    letter_paths.append(path + ",.png")
    letter_paths.append(path + "(.png")
    letter_paths.append(path + ").png")
    letter_paths.append(path + "SPACE.png")

    rows, cols = 5, 10

    fig, axes = plt.subplots(rows, cols)
    fig.suptitle(f"Alphabet Set: {alphabet[:-1].capitalize()}", fontsize = 16)
    axes = axes.flatten()

    mark = 0

    for i, image_path in enumerate(letter_paths):
        img = Image.open(image_path).convert("L")
        w, h = img.size
        pix = img.load()
        for y in range(h):
            for x in range(w):
                pix[x, y] = 255 - pix[x, y]
        axes[i].imshow(img, cmap = "gray")
        axes[i].axis("off")
        mark = i

    for j in range(mark + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    cos = np.abs(np.cos(angle))
    sin = np.abs(np.sin(angle))
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1.0)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated

def cut_image(image) :
    image_array = np.array(image)

    non_white_cols = np.any(np.any(image_array != [255, 255, 255], axis = -1), axis = 1)
    non_white_rows = np.any(np.any(image_array != [255, 255, 255], axis = -1), axis = 0)

    minimal_row = float('inf')
    maximal_row = float('-inf')
    minimal_col = float('inf')
    maximal_col = float('-inf')

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
    
    new_image_array = np.zeros((y_range, x_range, 3), dtype=np.uint8)
    
    for x in range(x_range):
        for y in range(y_range):
            new_image_array[y, x] = image_array[minimal_col + y, minimal_row + x]
    
    return new_image_array

def straighten_image(path: str):
    image = Image.open(path).convert("RGB")
    image = np.array(image)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    angles = []
    if lines is not None:
        for _, theta in lines[:, 0]:
            angles.append(theta - np.pi / 2)
    
    if angles: mean_angle = np.mean(angles)
    else: mean_angle = 0

    new_image = cut_image(rotate_image(image, mean_angle))
    straightened_image = cv2.fastNlMeansDenoisingColored(new_image, None, 10, 10, 7, 21)
    straightened_image = Image.fromarray(straightened_image).convert("L")

    pix = straightened_image.load()

    w, h = straightened_image.size

    for y in range(h):
        for x in range(w):
            pix[x, y] = 255 - pix[x, y]

    show_image(straightened_image, "Straightened image")
    straightened_image.save(path.replace("/", "/straightened_"))

    return straightened_image

alphabet_base_folder = "images_ocr/alphabets/"
chosen_font = "arial/"
text = "images_ocr/arial_text.png"

inverted_img_path = invert_image(text, chosen_font)
show_alphabet(chosen_font, alphabet_base_folder)
straighten_image(text)
