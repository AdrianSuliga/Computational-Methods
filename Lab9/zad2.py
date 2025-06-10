from PIL import Image
import matplotlib.pyplot as plt

def show_image(image, title):
    plt.imshow(image, cmap = "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def invert_image(path: str, font: str) -> None:
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

def show_alphabet(alphabet: str, folder: str) -> None:
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

    _, axes = plt.subplots(rows, cols)
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

alphabet_base_folder = "images_ocr/alphabets/"
chosen_font = "arial/"
text = "images_ocr/arial_text.png"

inverted_img_path = invert_image(text, chosen_font)
show_alphabet(chosen_font, alphabet_base_folder)