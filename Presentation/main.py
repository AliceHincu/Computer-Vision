import cv2
import matplotlib.pyplot as plt

def load_images():
    reference_img = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    if reference_img is None or test_img is None:
        print("One of the images was not loaded! Check path!")
        exit()

    return reference_img, test_img


def plot_result(result):
    plt.figure(figsize=(15, 10))
    plt.imshow(result)
    plt.title('ORB Feature Matching')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    reference_img, test_img = load_images()
    orb = cv2.ORB_create()

    # Detect keypoints and generate descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(reference_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_img, None)

    # Create Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between images
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches are the first ones)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches betwwen images
    result = cv2.drawMatches(reference_img, keypoints1, test_img, keypoints2, matches[:20], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plot_result(result)

