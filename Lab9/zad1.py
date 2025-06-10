from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def invert_img(path: str, folder_path: str, mode: str = "c") -> str:
    img = Image.open(f"{folder_path}{path}").convert("L")
    
    if mode == "c":
        pix = img.load()
        width, height = img.size

        for j in range(height):
            for i in range(width):
                pix[i, j] = 255 - pix[i, j]
    
    inverted_path = f"{folder_path}inverted_{path}"
    img.save(inverted_path)

    plt.imshow(img, cmap = "gray")
    plt.title(f"Odwrócony obraz o ścieżce {path}")
    plt.axis("off")
    plt.show()

    return inverted_path

def normalize_magnitude(fft_array):
    fft_array = np.log1p(fft_array)
    fft_array = 255 * fft_array / np.max(fft_array)
    return fft_array.astype(np.uint8)

def normalize_phase(fft_array):
    fft_array = (fft_array + np.pi) / (2 * np.pi)
    return (255 * fft_array).astype(np.uint8)

def transform(img_path: str) -> None:
    img = Image.open(img_path).convert("L")
    img_array = np.array(img)
    fft_img = np.fft.fft2(img_array)

    fft_img_shifted = np.fft.fftshift(fft_img)

    magnitude = np.abs(fft_img_shifted)
    magnitude = normalize_magnitude(magnitude)

    phase = np.angle(fft_img_shifted)
    phase = normalize_phase(phase)

    magnitude_image = Image.fromarray(magnitude)
    phase_image = Image.fromarray(phase)

    magnitude_path = img_path.replace("inverted", "moduł")
    phase_path = img_path.replace("inverted", "faza")

    magnitude_image.save(magnitude_path)
    phase_image.save(phase_path)

    plt.imshow(magnitude_image, cmap="gray")
    plt.title("Wartości modułu współczynników Fouriera (środek = DC)")
    plt.axis("off")
    plt.show()

    plt.imshow(phase_image, cmap="gray")
    plt.title("Wartości fazy współczynników Fouriera (środek = DC)")
    plt.axis("off")
    plt.show()

def corelation(img_path: str, pattern_path: str) -> None:
    img = Image.open(img_path).convert("L")
    pattern = Image.open(pattern_path).convert("L")

    w, h = img.size

    cor = np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(np.rot90(np.rot90(pattern)), s = (h, w))).real
    cor /= np.abs(np.max(cor))

    return cor

def show_corelation(cor):
    cor -= np.min(cor)
    cor = 255 * cor / np.max(cor)
    plt.imshow(cor, cmap = "gray")
    plt.title("Wyniki korelacji")
    plt.axis("off")
    plt.show()

def mark_matches(img_path: str, pattern_path: str, corelation, threshold: float) -> None:
    img = Image.open(img_path).convert("RGB")
    pattern = Image.open(pattern_path).convert("RGB")

    maxi = -1
    for c in corelation:
        for val in c:
            maxi = max(maxi, val)

    padding = 0
    pw, ph = pattern.size
    draw = ImageDraw.Draw(img)

    for y in range(corelation.shape[0]):
        for x in range(corelation.shape[1]):
            if corelation[y, x] >= threshold:
                draw.rectangle(
                    [x - pw - padding, y - ph - padding, x + padding, y + padding],
                    outline = "red",
                    width = 1
                )
    
    plt.imshow(img)
    plt.title("Znalezione wzorce")
    plt.axis("off")
    plt.show()

img_folder = "images_ex1/"
img_path = "fish.png"
pattern_path = "fish_m.png"

inverted_img_path = invert_img(img_path, img_folder, "f")
inverted_pattern_path = invert_img(pattern_path, img_folder, "f")
transform(inverted_img_path)
cor = corelation(inverted_img_path, inverted_pattern_path)
show_corelation(cor)
mark_matches(inverted_img_path, inverted_pattern_path, cor, 0.88)