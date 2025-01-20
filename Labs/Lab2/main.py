"""
Documentation:
This script overlays a logo with transparency onto a video.

### How Alpha Blending Works:
1. The logo image is loaded with transparency (BGRA format).
2. The alpha channel is extracted and converted to a 3-channel grayscale image.
3. The alpha channel is normalized to values between [0,1] for blending.
4. Each video frame is processed to overlay the logo without hiding transparent areas.

### Example:
If we have a BGRA pixel:
    [B, G, R, A] = [120, 200, 50, 128]
Here:
    - B = 120 (Blue)
    - G = 200 (Green)
    - R = 50 (Red)
    - A = 128 (50% transparent)

After extracting alpha (which is 2D):
    alpha = [[128, 255, 0],
             [100, 50, 200],
             [30, 220, 180]]

Converted to BGR for blending (converting to 3D):
    alpha = [[[128, 128, 128], [255, 255, 255], [0, 0, 0]],
             [[100, 100, 100], [50, 50, 50], [200, 200, 200]],
             [[30, 30, 30], [220, 220, 220], [180, 180, 180]]]

Normalized (divided by 255.0):
    alpha = [[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
             [[0.39, 0.39, 0.39], [0.20, 0.20, 0.20], [0.78, 0.78, 0.78]],
             [[0.12, 0.12, 0.12], [0.86, 0.86, 0.86], [0.71, 0.71, 0.71]]]

### Blending Formula:
    result = (background * (1 - alpha) + foreground * alpha)

This ensures that transparent parts of the logo do not completely hide the video background.
"""

import cv2
import numpy as np

logo_name = 'opencv_logo.png'
video_name = 'Megamind.avi'

def get_color_channels(logo):
    if logo.shape[2] == 4:
        return cv2.split(logo)
    else:
        print("Logo should have alpha channel!")
        exit()

def get_logo_resized_and_alpha(logo):
    """
    Loads a logo image, resizes it, extracts the alpha channel, and prepares it for blending.

    Parameters:
    logo_name (str): Path to the logo image file.

    Returns:
    tuple: (logo_resized, alpha) where:
        - logo_resized (numpy.ndarray): The resized logo image without alpha channel.
        - alpha (numpy.ndarray): The normalized alpha mask (values between [0,1]).
    """
    logo = cv2.imread(logo_name, cv2.IMREAD_UNCHANGED)
    logo_resized = cv2.resize(logo, (150, 150))
    b, g, r, alpha = cv2.split(logo_resized)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) / 255.0
    logo_resized = cv2.merge([b, g, r])
    return logo_resized, alpha

def apply_alpha_blending(logo, alpha):
    """
    roi * (1 - alpha) → Keep video pixels in the transparent zones of the logos
    logo_resized * alpha → Show logo in the places where alpha is bigger
    """
    return (roi * (1 - alpha) + logo * alpha).astype(np.uint8) # alpha blending


if __name__ == '__main__':
    logo_resized, alpha = get_logo_resized_and_alpha(logo_name)

    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print("Could not open video file!")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[10:160, 10:160] # region of interest where logo is placed
        blended = apply_alpha_blending(logo_resized, alpha)
        frame[10:160, 10:160] = blended
        cv2.imshow('Video cu Logo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
