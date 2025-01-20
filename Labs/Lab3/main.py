import cv2
import numpy as np


def morphological_corner_extraction(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert to binary
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Define structuring elements
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    diamond = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    x_shape = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply morphological operations
    R1 = cv2.dilate(binary, cross)
    R1 = cv2.erode(R1, diamond)

    R2 = cv2.dilate(binary, x_shape)
    R2 = cv2.erode(R2, square)

    # Compute absolute difference
    R = cv2.absdiff(R2, R1)

    # Overlay on original image
    color_img = cv2.imread(image_path)
    contours, _ = cv2.findContours(R, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_img, contours, -1, (0, 0, 255), 2)

    # Display results
    cv2.imshow("Extracted Corners", R)
    cv2.imshow("Original Image with Corners", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Apply to images
morphological_corner_extraction("Rectangle.png")
morphological_corner_extraction("Building.png")
