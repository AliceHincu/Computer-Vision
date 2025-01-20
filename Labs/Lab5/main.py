import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_edges(gray, lower_bound_threshold=100, upper_bound_threshold=200):
    """
    Detects edges in a grayscale image using the Canny edge detection method.
    Returns: Edge-detected image.
    """
    return cv2.Canny(gray, lower_bound_threshold, upper_bound_threshold)


def detect_lines(image, edges, rho=1, theta=np.pi / 180, threshold=65):
    """
    Detects lines in an image using the Hough Transform.

    Parameters:
        image: The original image on which lines will be drawn.
        edges: The edge-detected image.
        rho (float): Distance resolution of the accumulator in pixels.
        theta (float): Angle resolution of the accumulator in radians.
        threshold (int): The minimum number of votes required to consider a line.

    Explanation:
        - The Hough Transform detects lines in polar coordinates (rho, theta).
        - (rho, theta) are converted into Cartesian coordinates.
        - The line direction is calculated using cos(theta) and sin(theta).
        - To extend the detected line, we compute two points in opposite directions:
            - The normal direction of the line is (-sin(theta), cos(theta)).
            - `x1, y1` and `x2, y2` are computed by moving along this direction.
        - The factor `1000` ensures the line is long enough to cover the image.

    Returns:
        None (draws detected lines on the image directly).
    """
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # Convert polar coordinates (rho, theta) to Cartesian coordinates
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho

            # Determine two endpoints of the detected line to draw it on the image
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))

            # Draw the line on the image (green color, thickness=2)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def detect_segments(image, edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10):
    """
    Detects line segments in an image using the Probabilistic Hough Transform.

    Parameters:
        image (numpy.ndarray): The original image on which segments will be drawn.
        edges (numpy.ndarray): The edge-detected image.
        rho (float): Distance resolution of the accumulator in pixels.
        theta (float): Angle resolution of the accumulator in radians.
        threshold (int): Minimum number of votes required to consider a segment.
        minLineLength (int): Minimum length of a line segment.
        maxLineGap (int): Maximum gap between segments to be considered the same line.

    Returns:
        None (draws detected segments on the image directly).
    """
    segments = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if segments is not None:
        for segment in segments:
            x1, y1, x2, y2 = segment[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines for segments


def detect_circles(image, gray, dp=1.5, minDist=50, param1=100, param2=50, minRadius=20, maxRadius=100):
    """
    Detects circles in an image using the Hough Gradient Method.

    Parameters:
        image : The original image on which circles will be drawn.
        gray : Grayscale version of the image.
        dp (float): Inverse ratio of the accumulator resolution to the image resolution.
        minDist (int): Minimum distance between detected circle centers.
        param1 (int): Upper threshold for the Canny edge detector.
        param2 (int): Threshold for center detection (higher means fewer detections).
        minRadius (int): Minimum circle radius.
        maxRadius (int): Maximum circle radius.

    Returns:
        None (draws detected circles on the image directly).
    """
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius,
                               maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle in red
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)

            # Draw the center of the circle in blue
            cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 3)


def main():
    """
    Main function to load an image, process it, and detect edges, lines, and circles.
    """
    image = cv2.imread('image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray)

    detect_lines(image, edges)
    detect_segments(image, edges)
    detect_circles(image, gray)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
