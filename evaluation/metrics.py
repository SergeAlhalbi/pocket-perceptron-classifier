import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """
    A class to compute various evaluation metrics for classification.
    """

    @staticmethod
    def compute_confusion_matrix(Y_true, Y_pred, class_labels):
        """
        Computes a confusion matrix for a given set of true and predicted labels.

        Args:
            Y_true (numpy.ndarray): Ground truth labels.
            Y_pred (numpy.ndarray): Predicted labels.
            class_labels (list): Unique class labels.

        Returns:
            numpy.ndarray: Confusion matrix of shape (num_classes, num_classes).
        """
        num_classes = len(class_labels)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Create a mapping from class label to index
        label_to_index = {label: idx for idx, label in enumerate(class_labels)}

        for true_label, pred_label in zip(Y_true, Y_pred):
            i = label_to_index[true_label]  # Row index (Actual class)
            j = label_to_index[pred_label]  # Column index (Predicted class)
            conf_matrix[i, j] += 1  # Increment count

        return conf_matrix

    @staticmethod
    def plot_confusion_matrix(conf_matrix, class_labels, title="Confusion Matrix"):
        """
        Plots a heatmap for the confusion matrix.

        Args:
            conf_matrix (numpy.ndarray): Confusion matrix.
            class_labels (list): Labels for the classes.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(7, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(title)
        plt.show()

    @staticmethod
    def overall_accuracy(Y_true, Y_pred):
        """
        Computes overall accuracy.

        Args:
            Y_true (numpy.ndarray): Ground truth labels.
            Y_pred (numpy.ndarray): Predicted labels.

        Returns:
            float: Overall accuracy.
        """
        return np.mean(Y_true == Y_pred)
    
    @staticmethod
    def class_accuracy(conf_matrix):
        """
        Computes accuracy per class.

        Args:
            conf_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            numpy.ndarray: Accuracy for each class.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            acc = np.nan_to_num(acc)
        return acc

    @staticmethod
    def producer_accuracy(conf_matrix):
        """
        Computes Producer’s Accuracy (Recall per class).

        Args:
            conf_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            numpy.ndarray: Producer’s accuracy for each class.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            recall = np.nan_to_num(recall)  # Convert NaNs to 0
        return recall

    @staticmethod
    def user_accuracy(conf_matrix):
        """
        Computes User’s Accuracy (Precision per class).

        Args:
            conf_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            numpy.ndarray: User’s accuracy for each class.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
            precision = np.nan_to_num(precision)  # Convert NaNs to 0
        return precision

    @staticmethod
    def kappa_coefficient(conf_matrix):
        """
        Computes Kappa Coefficient to measure classification agreement.

        Args:
            conf_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            float: Kappa coefficient.
        """
        total = np.sum(conf_matrix)
        sum_po = np.sum(np.diag(conf_matrix))  # Observed agreement
        sum_pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / total  # Expected agreement
        return (sum_po - sum_pe) / (total - sum_pe) if (total - sum_pe) != 0 else 0

# Example Usage
if __name__ == "__main__":
    # Dummy data: Ground truth vs Predicted labels (multi-class example)
    Y_true = np.array([0, 1, 2, 1, 0, 2, 1, 2, 0, 1])
    Y_pred = np.array([0, 1, 2, 0, 0, 2, 2, 2, 1, 1])
    class_labels = [0, 1, 2]  # Unique classes

    # Compute confusion matrix
    conf_matrix = Evaluator.compute_confusion_matrix(Y_true, Y_pred, class_labels)
    Evaluator.plot_confusion_matrix(conf_matrix, class_labels, title="Multi-Class Confusion Matrix")

    # Compute metrics
    accuracy = Evaluator.overall_accuracy(Y_true, Y_pred)
    class_accuracy = Evaluator.class_accuracy(conf_matrix)
    producer_acc = Evaluator.producer_accuracy(conf_matrix)
    user_acc = Evaluator.user_accuracy(conf_matrix)
    kappa = Evaluator.kappa_coefficient(conf_matrix)

    print("Confusion Matrix:\n", conf_matrix)
    print("Overall Accuracy:", accuracy)
    print("Class Accuracy:", class_accuracy)
    print("Producer’s Accuracy (Recall):", producer_acc)
    print("User’s Accuracy (Precision):", user_acc)
    print("Kappa Coefficient:", kappa)