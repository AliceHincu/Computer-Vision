import cv2
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in glob.glob(os.path.join(folder, "*.jpg")):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(filename))
    return images, filenames


def compute_histogram(image, bins=256):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
    hist = cv2.calcHist([image_hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])  # Only H and S channels
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def compare_histograms(query_hist, histograms, methods):
    results = {}
    for method in methods:
        similarity_scores = [cv2.compareHist(query_hist, h, method) for h in histograms]

        # Normalize each metric separately
        min_val, max_val = min(similarity_scores), max(similarity_scores)
        normalized_scores = [(s - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0 for s in
                             similarity_scores] # make sure it's independent of iage dimension

        results[method] = normalized_scores
    return results


# Load images
data_folder = "images"  # Folder where images are stored
images, filenames = load_images_from_folder(data_folder)
if len(images) < 20:
    raise ValueError("Please provide at least 20 images in the dataset.")

# Select query image by filename
query_filename = "1.jpg"  # Change this to the desired image filename
if query_filename not in filenames:
    raise ValueError(f"Query image {query_filename} not found in dataset.")
query_index = filenames.index(query_filename)

# Compute histograms
histograms_256 = [compute_histogram(img, bins=256) for img in images]
histograms_64 = [compute_histogram(img, bins=64) for img in images]
histograms_32 = [compute_histogram(img, bins=32) for img in images]

# Select query image histogram
query_hist_256 = histograms_256[query_index]
query_hist_64 = histograms_64[query_index]
query_hist_32 = histograms_32[query_index]

# Define comparison methods
methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
method_names = ["Correlation", "Chi-Square", "Intersection", "Bhattacharyya"]

# Compare histograms
results_256 = compare_histograms(query_hist_256, histograms_256, methods)
results_64 = compare_histograms(query_hist_64, histograms_64, methods)
results_32 = compare_histograms(query_hist_32, histograms_32, methods)


# Display results
def display_results(results, bins):
    df = pd.DataFrame(results, index=filenames)
    df.index.name = "Image"
    df.columns = method_names
    print(f"\nResults for {bins} bins:")
    print(df)
    return df


df_256 = display_results(results_256, 256)
df_64 = display_results(results_64, 64)
df_32 = display_results(results_32, 32)

# Save results
df_256.to_csv("histogram_comparison_256.csv")
df_64.to_csv("histogram_comparison_64.csv")
df_32.to_csv("histogram_comparison_32.csv")


# Compare visualization
def plot_histogram_comparisons(df, bins):
    df.plot(kind='bar', figsize=(12, 6), title=f"Histogram Similarity Scores - {bins} bins")
    plt.xlabel("Image")
    plt.ylabel("Normalized Similarity Score")
    plt.legend(title="Methods")
    plt.xticks(rotation=90)
    plt.show()


plot_histogram_comparisons(df_256, 256)
plot_histogram_comparisons(df_64, 64)
plot_histogram_comparisons(df_32, 32)

"""
- Correlation & Intersection -> higher values = similar images
- Chi-Square & Bhattacharyya -> lower values = similar images.
- Reducing the number of bins (from 256 → 64 → 32) affects the ranking slightly but maintains general trends.


Interpretation of Metrics
- Correlation (Higher = More Similar)
  - Images 4.jpg, 6.jpg, and 18.jpg appear the most similar.
- Chi-Square (Lower = More Similar)
  - Images 4.jpg, 6.jpg, and 18.jpg match closely.
- Intersection (Higher = More Similar)
  - 4.jpg and 6.jpg score well.
- Bhattacharyya (Lower = More Similar)
  - 4.jpg and 6.jpg again stand out as similar.

Conclusion
- Images 4.jpg and 6.jpg are the most similar to the query image.
- Reducing the bins makes similarity detection more focused on major color trends.
- Histogram comparison is effective for color similarity but not for shape or object recognition.
"""