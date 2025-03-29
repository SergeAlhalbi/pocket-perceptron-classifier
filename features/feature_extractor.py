from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    """
    A class to extract features from images, such as symmetry and intensity.
    """
    
    def __init__(self, images):
        """
        Initializes the FeatureExtractor with the images.

        Args:
            images (numpy.ndarray): Array of images (shape: [num_samples, height, width]).
        """
        self.images = images
    
    def compute_symmetry(self, image):
        """
        Computes symmetry as the mean absolute difference between the left and right halves of the image.

        Args:
            image (numpy.ndarray): A single grayscale image.
        
        Returns:
            float: Symmetry score (lower means more symmetrical).
        """
        mid = image.shape[1] // 2
        left_half = image[:, :mid]
        right_half = np.fliplr(image[:, mid:])  # Flip right half horizontally
        symmetry_score = np.mean(np.abs(left_half - right_half))
        return symmetry_score
    
    def compute_intensity(self, image):
        """
        Computes the average pixel intensity of an image.

        Args:
            image (numpy.ndarray): A single grayscale image.
        
        Returns:
            float: Average intensity value.
        """
        return np.mean(image)
    
    def extract_features(self, use_symmetry=True, use_intensity=True, use_hog=True):
        """
        Extracts selected features (symmetry, intensity, and/or HOG) for all images.

        Args:
            use_symmetry (bool): Whether to include symmetry features.
            use_intensity (bool): Whether to include intensity features.
            use_hog (bool): Whether to include HOG features.

        Returns:
            numpy.ndarray: Feature matrix (num_samples, num_selected_features).
        """
        feature_list = []

        if use_symmetry:
            symmetry_features = [self.compute_symmetry(img) for img in self.images]
            feature_list.append(np.array(symmetry_features).reshape(-1, 1))

        if use_intensity:
            intensity_features = [self.compute_intensity(img) for img in self.images]
            feature_list.append(np.array(intensity_features).reshape(-1, 1))

        if use_hog:
            hog_features = [self.compute_hog(img) for img in self.images]
            feature_list.append(np.array(hog_features))

        # Combine all selected features
        features = np.hstack(feature_list)
        return features

    
    def compute_hog(self, image, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(2, 2), show_image=False):
        """
        Computes HOG (Histogram of Oriented Gradients) features for a single image.

        Args:
            image (numpy.ndarray): A single grayscale image.
            orientations (int): Number of orientation bins.
            pixels_per_cell (tuple): Size (in pixels) of a cell.
            cells_per_block (tuple): Number of cells in each block.

        Returns:
            numpy.ndarray: 1D HOG feature vector.
        """
        features, hog_image = hog(image,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm='L2-Hys',
                       visualize=True,
                       feature_vector=True)
        # Show HOG image
        if show_image:
            plt.imshow(hog_image, cmap='gray')
            plt.title("HOG Visualization")
            plt.axis('off')
            plt.show()
        return features
    
    def add_bias(self, X_p):
        """
        Adds a bias term (column of ones) to the feature matrix.
        
        Args:
            X_p (numpy.ndarray): Feature matrix of shape (num_samples, num_features).
        
        Returns:
            X (numpy.ndarray): Feature matrix with bias added (num_samples, num_features + 1).
        """
        return np.c_[np.ones(X_p.shape[0]), X_p]

# Example Usage
if __name__ == "__main__":
    # Dummy data for testing
    sample_images = np.random.rand(5, 20, 20)  # 5 random grayscale images of size 20x20
    
    extractor = FeatureExtractor(sample_images)
    extracted_features_set_1 = extractor.extract_features(use_symmetry=True, use_intensity=True, use_hog=False)
    extracted_features_set_1 = extractor.add_bias(extracted_features_set_1)
    print("Extracted features shape:", extracted_features_set_1.shape)
    print("First few extracted features:\n", extracted_features_set_1[:5])
    
    extracted_features_set_2 = extractor.extract_features(use_symmetry=False, use_intensity=False, use_hog=True)
    extracted_features_set_2 = extractor.add_bias(extracted_features_set_2)
    print("Extracted features shape:", extracted_features_set_2.shape)
    print("First few extracted features:\n", extracted_features_set_2[:5])