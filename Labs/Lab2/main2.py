import cv2
import numpy as np

# Read the image in grayscale
image = cv2.imread('HappyFish.jpg', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding to convert the image into a black-and-white format
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Create a mask for flood fill operation
h, w = binary.shape
mask = np.zeros((h+2, w+2), np.uint8)

# Copy the binary image to apply flood fill
flood_fill = binary.copy()

# Perform the flood fill operation starting from the top-left corner (assuming the background is black)
cv2.floodFill(flood_fill, mask, (0, 0), 255)

# Invert the flood-filled image to obtain only the filled interior of the contour
flood_fill_inv = cv2.bitwise_not(flood_fill)

# Combine the original binary image with the inverted flood-filled image to get the fully filled contour
filled_image = binary | flood_fill_inv

# Display the result
cv2.imshow("Filled Contour", filled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

