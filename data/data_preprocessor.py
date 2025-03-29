import numpy as np

class DataPreprocessor:
    """
    A class for handling data preprocessing tasks such as normalization, shuffling, and adding bias.
    """

    def __init__(self, seed=42):
        """
        Initializes the DataPreprocessor with a fixed random seed.
        
        Args:
            seed (int): Random seed for reproducibility.
        """
        self.seed = seed
        np.random.seed(seed)

    def normalize(self, X):
        """
        Normalizes image pixel values to the range [0, 1].
        
        Args:
            X (numpy.ndarray): Feature matrix (num_samples, num_features).
        
        Returns:
            numpy.ndarray: Normalized feature matrix.
        """
        return X / 255.0

    def shuffle(self, X, Y):
        """
        Shuffles the dataset while keeping images and labels paired.
        
        Args:
            X (numpy.ndarray): Feature matrix.
            Y (numpy.ndarray): Corresponding labels.
        
        Returns:
            tuple: Shuffled (X, Y).
        """
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        return X[indices], Y[indices]

# Example Usage
if __name__ == "__main__":
    # Dummy data
    X_train = np.random.rand(100, 2) * 255  # Simulating raw pixel intensity values
    Y_train = np.random.choice([-1, 1], size=100)  # Binary labels
    
    preprocessor = DataPreprocessor()
    
    X_train = preprocessor.normalize(X_train)
    X_train, Y_train = preprocessor.shuffle(X_train, Y_train)

    print("Preprocessed Data Shape:", X_train.shape)