from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("lena_gray.png").convert("L")
I = np.asarray(image)

U, S, Vt = np.linalg.svd(I, full_matrices = False)

ks = [1, 2, 5, 10, 15, 20, 30, 40, 50, 100]
ns = []

for k in ks:
    IA = S[:k] * U[:, :k] @ Vt[:k, :]
    ns.append(np.linalg.norm(I - IA))
    plt.imshow(IA, cmap='gray')
    plt.axis('off')
    plt.title(f'k = {k}')
    plt.show()  

plt.plot(ks, ns)
plt.show()