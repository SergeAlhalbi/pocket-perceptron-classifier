import os
import numpy as np
import cv2

class DataLoader:
    """
    A class to load, normalize, and shuffle image data for training and testing.

    Attributes:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the testing dataset.
    """

    def __init__(self, train_path, test_path):
        """
        Initializes the DataLoader with paths to the training and testing datasets.

        Args:
            train_path (str): Path to the training data.
            test_path (str): Path to the testing data.
        """
        self.train_path = train_path
        self.test_path = test_path
    
    def load_images(self, path):
        """
        Loads grayscale images from the specified directory.

        Args:
            path (str): Directory path containing image subfolders (each subfolder is a class label).

        Returns:
            tuple: A tuple containing:
                - images (numpy.ndarray): Array of loaded images (shape: [num_samples, height, width]).
                - labels (numpy.ndarray): Corresponding labels for each image.
        """
        images = []
        labels = []
        
        for label in sorted(os.listdir(path)):
            label_path = os.path.join(path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                    images.append(img)
                    labels.append(int(label))  # Folder name is the label
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return images, labels
    
    
    
    def get_train_test_data(self):
        """
        Loads and returns the training and testing datasets.

        Returns:
            tuple: A tuple containing:
                - X_train (numpy.ndarray): Training images.
                - X_test (numpy.ndarray): Testing images.
                - Y_train (numpy.ndarray): Training labels.
                - Y_test (numpy.ndarray): Testing labels.
        """
        X_train, Y_train = self.load_images(self.train_path)
        X_test, Y_test = self.load_images(self.test_path)
        return X_train, X_test, Y_train, Y_test

# Example Usage
if __name__ == "__main__":
    # Paths
    TRAINING_PATH = r"C:\Users\serge\OneDrive\Desktop\Education\OSU\Classes\Advanced Machine Learning for Remote Sensing\Assignments\HW1\Work\data\training"
    TESTING_PATH = r"C:\Users\serge\OneDrive\Desktop\Education\OSU\Classes\Advanced Machine Learning for Remote Sensing\Assignments\HW1\Work\data\testing"
    
    data_loader = DataLoader(TRAINING_PATH, TESTING_PATH)
    X_train, X_test, Y_train, Y_test = data_loader.get_train_test_data()
    
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Training labels shape:", Y_train.shape)
    print("Testing labels shape:", Y_test.shape)