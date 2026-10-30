import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('origi.png', 0)

# Define the structuring element for the pattern to be detected (0 for don't care, 1 for foreground, -1 for background)
pattern = np.array([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]], dtype=np.int8)

# Apply the Hit or Miss Transformation
result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, pattern)

# Display the original image and the result using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Hit-or-Miss Result')
plt.axis('off')

plt.tight_layout()
plt.show()
