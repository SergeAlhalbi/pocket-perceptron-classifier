import numpy as np
import matplotlib.pyplot as plt

class PocketPerceptron:
    """
    Implementation of the Pocket Algorithm for binary classification.
    """
    
    def __init__(self, num_features, max_iterations=25000, learning_rate=.005, tolerance=1e-4, patience=10, seed=42):
        """
        Initializes the Pocket Perceptron with random weights.
        
        Args:
            num_features (int): Number of input features.
            max_iterations (int): Maximum number of training iterations.
            learning_rate (float): Step size for weight updates.
            tolerance (float): Minimum change in accuracy to continue training.
            patience (int): Number of iterations to wait before stopping if no improvement.
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.weights = np.random.randn(num_features + 1)  # +1 for bias
        self.pocket_weights = self.weights.copy()
        self.best_accuracy = 0.0
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.patience = patience
    
    def predict(self, X, W, return_scores=False):
        """
        Computes predicted class labels or raw scores.

        Args:
            X (numpy.ndarray): Input features (num_samples, num_features).
            W (numpy.ndarray): Weight vector used for classification.
            return_scores (bool): If True, returns raw scores instead of class labels.

        Returns:
            numpy.ndarray: 
                - If return_scores=True: Raw dot product scores.
                - If return_scores=False: Predicted class labels (+1 or -1).
        """
        raw_scores = np.dot(X, W)  # Compute raw scores
    
        if return_scores:
            return raw_scores  # Return raw values for multi-class ranking

        return np.where(np.sign(raw_scores) == 0, 1, np.sign(raw_scores))  # Convert 0 to 1

    def evaluate(self, X, Y, W):
        """
        Evaluates the accuracy of the current model.
        
        Args:
            X (numpy.ndarray): Feature matrix.
            Y (numpy.ndarray): True labels.
            W (numpy.ndarray): Weight vector used for classification.
        
        Returns:
            float: Accuracy score.
        """
        predictions = self.predict(X, self.weights)
        return np.mean(predictions == Y)
    
    def train(self, X, Y, verbose=False, print_every=500):
        """
        Trains the Pocket Perceptron using the given dataset.
        
        Args:
            X (numpy.ndarray): Training features (num_samples, num_features).
            Y (numpy.ndarray): Training labels (+1 or -1).
            verbose (bool): If True, prints progress every `print_every` iterations.
            print_every (int): Determines how often to print progress.
        """
        tolerance_check = False
        
        for i in range(self.max_iterations):
            misclassified = np.where(self.predict(X, self.weights) != Y)[0]  # Find misclassified samples
            
            if len(misclassified) == 0:
                if verbose:
                    print(f" Converged after {i} iterations.")
                break  # Stop early if no errors
            
            # Pick a random misclassified sample
            idx = np.random.choice(misclassified)
            
            # Update weights using perceptron rule
            self.weights += self.learning_rate * Y[idx] * X[idx]
            
            # Evaluate new accuracy
            accuracy = self.evaluate(X, Y, self.weights)
            
            # Update pocket weights if performance improves
            if accuracy > self.best_accuracy + self.tolerance:
                self.best_accuracy = accuracy
                self.pocket_weights = self.weights.copy()
                no_improve_count = 0  # Reset patience counter
            else:
                no_improve_count += 1  # Increment patience counter
                
            # Print progress every `print_every` iterations
            if verbose and (i + 1) % print_every == 0:
                print(f"Iteration {i + 1}/{self.max_iterations} - Best Accuracy: {self.best_accuracy:.4f}")
            
            # Stop early if no improvement for `patience` iterations
            if tolerance_check == True:
                if no_improve_count >= self.patience:
                    if verbose:
                        print(f"Stopping early at iteration {i}, no improvement for {self.patience} iterations.")
                        break
                
    def plot_decision_boundary(perceptron, X, Y):
        """
        Plots the decision boundary of a trained Pocket Perceptron model.

        Args:
            perceptron (PocketPerceptron): Trained perceptron model.
            X (numpy.ndarray): Feature matrix (num_samples, num_features).
            Y (numpy.ndarray): Class labels (+1 or -1).

        Raises:
        ValueError: If the number of features is not 3 (including bias term).
        """
        if X.shape[1] != 3:
            raise ValueError(f"Error: Expected input with 3 features (including bias), but got shape {X.shape}. "
                             "Ensure bias is added before calling this function.")

        plt.figure(figsize=(6, 5))

        # Scatter plot of the data points
        plt.scatter(X[Y == 1, 1], X[Y == 1, 2], marker='.', label="Class 1", edgecolors='black', color="blue")
        plt.scatter(X[Y == -1, 1], X[Y == -1, 2], marker='.', label="Class -1", edgecolors='black', color="red")

        # Decision boundary line (W0 + W1*x1 + W2*x2 = 0)
        x_min, x_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        x1_vals = np.linspace(x_min, x_max, 100)
        w = perceptron.pocket_weights

        if w[2] == 0:
            raise ValueError("Error: The second feature weight (w[2]) is zero, making the decision boundary undefined.")

        x2_vals = - (w[0] + w[1] * x1_vals) / w[2]  # Solve for x2

        plt.plot(x1_vals, x2_vals, 'k--', label="Decision Boundary")
        
        # Fix axis limits
        plt.xlim(0, 1.2)
        plt.ylim(0, 1.2)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary")
        plt.legend()
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Dummy data for testing
    X_train = np.random.rand(100, 2)  # 100 samples, 2 features
    X_train = np.c_[np.ones(X_train.shape[0]), X_train] # Add bias manually
    Y_train = np.random.choice([-1, 1], size=100)  # Binary labels (-1, 1)
    
    perceptron = PocketPerceptron(num_features=2)
    perceptron.train(X_train, Y_train, True)
    
    training_predictions_scores = perceptron.predict(X_train, perceptron.pocket_weights, True)
    training_predictions = perceptron.predict(X_train, perceptron.pocket_weights)
    training_accuracy = perceptron.evaluate(X_train, Y_train, perceptron.pocket_weights)
    
    print("Training Accuracy with Pocket Algorithm:", training_accuracy)
    
    perceptron.plot_decision_boundary(X_train, Y_train)