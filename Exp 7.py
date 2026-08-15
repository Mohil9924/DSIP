import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Load Image
# Replace 'image.jpg' with your actual image filename or full path
# Example path: r'C:\Users\MOHIL\Desktop\my_image.jpg'
# ---------------------------------------------------------
image_path = "D:\images.jpg"
src_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image was loaded properly
if src_image is None:
    print(f"Error: Could not load image from '{image_path}'. Check the path!")
else:
    # -------------------------
    # 1. Image Negation
    # -------------------------
    negative_image = 255 - src_image

    # -------------------------
    # 2. Thresholding
    # -------------------------
    _, thresholded_image = cv2.threshold(
        src_image, 128, 255, cv2.THRESH_BINARY
    )

    # -------------------------
    # 3. Gamma Correction (gamma = 2.0 brightens mid-tones)
    # -------------------------
    gamma = 2.0
    normalized_image = src_image / 255.0
    gamma_corrected_image = np.power(normalized_image, 1 / gamma)
    gamma_corrected_image = np.uint8(gamma_corrected_image * 255)

    # -------------------------
    # Display Results in Spyder
    # -------------------------
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(src_image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(negative_image, cmap="gray")
    plt.title("Negative")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(thresholded_image, cmap="gray")
    plt.title("Threshold")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(gamma_corrected_image, cmap="gray")
    plt.title("Gamma Corrected")
    plt.axis("off")

    plt.tight_layout()
    plt.show()