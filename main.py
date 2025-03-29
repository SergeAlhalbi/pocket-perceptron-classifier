from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from features.feature_extractor import FeatureExtractor
from models.pocket_perceptron import PocketPerceptron
from evaluation.metrics import Evaluator
from methods.multi_class_strategy import one_vs_all, one_vs_one

from pathlib import Path
import numpy as np
from itertools import combinations

# Define base directory and data paths (relative)
BASE_DIR = Path(__file__).resolve().parent
TRAINING_PATH = BASE_DIR / "data" / "training"
TESTING_PATH = BASE_DIR / "data" / "testing"

def run_experiment(feature_set, mode):
    """
    Runs a digit classification experiment using the selected feature set and multi-class strategy.

    Args:
        feature_set (int): 
            1 for symmetry + intensity features, 
            2 for HOG features.
        
        mode (str): 
            The multi-class strategy to use:
            - "one_vs_all" for One-vs-All classification
            - "one_vs_one" for One-vs-One classification
    """
    num_classes = 10

    # Load data
    data_loader = DataLoader(TRAINING_PATH, TESTING_PATH)
    X_train, X_test, Y_train, Y_test = data_loader.get_train_test_data()

    if feature_set == 1:
        num_features = 2
        preprocessor = DataPreprocessor()
        X_train = preprocessor.normalize(X_train)
        X_train, Y_train = preprocessor.shuffle(X_train, Y_train)
        X_test = preprocessor.normalize(X_test)
        X_test, Y_test = preprocessor.shuffle(X_test, Y_test)

        extractor_train = FeatureExtractor(X_train)
        X_train = extractor_train.extract_features(use_symmetry=True, use_intensity=True, use_hog=False)
        X_train = extractor_train.add_bias(X_train)

        extractor_test = FeatureExtractor(X_test)
        X_test = extractor_test.extract_features(use_symmetry=True, use_intensity=True, use_hog=False)
        X_test = extractor_test.add_bias(X_test)

    elif feature_set == 2:
        num_features = 324
        extractor_train = FeatureExtractor(X_train)
        X_train = extractor_train.extract_features(use_symmetry=False, use_intensity=False, use_hog=True)
        X_train = extractor_train.add_bias(X_train)

        extractor_test = FeatureExtractor(X_test)
        X_test = extractor_test.extract_features(use_symmetry=False, use_intensity=False, use_hog=True)
        X_test = extractor_test.add_bias(X_test)

    if mode == "one_vs_all":
        predictions_train = {}
        predictions_test = {}

        for target_digit in range(num_classes):
            print(f"\n Training One-vs-All for Digit {target_digit}...")
            Y_train_bin = one_vs_all(Y_train, target_digit)
            Y_test_bin = one_vs_all(Y_test, target_digit)

            perceptron = PocketPerceptron(num_features)
            perceptron.train(X_train, Y_train_bin, True)

            predictions_train[target_digit] = perceptron.predict(X_train, perceptron.pocket_weights, True)
            predictions_test[target_digit] = perceptron.predict(X_test, perceptron.pocket_weights, True)

            if num_features == 2:
                PocketPerceptron.plot_decision_boundary(perceptron, X_train, Y_train_bin)

        Y_pred_final_train = np.argmax(np.array(list(predictions_train.values())).T, axis=1)
        Y_pred_final_test = np.argmax(np.array(list(predictions_test.values())).T, axis=1)

    elif mode == "one_vs_one":
        ovo_predictions_train = {}
        ovo_predictions_test = {}
        final_votes_train = np.zeros((len(Y_train), num_classes))
        final_votes_test = np.zeros((len(Y_test), num_classes))

        for class_1, class_2 in combinations(range(num_classes), 2):
            print(f"\n Training One-vs-One for {class_1} vs {class_2}...")

            X_train_ovo, Y_train_ovo, train_indices = one_vs_one(X_train, Y_train, class_1, class_2)
            X_test_ovo, Y_test_ovo, test_indices = one_vs_one(X_test, Y_test, class_1, class_2)

            perceptron = PocketPerceptron(num_features)
            perceptron.train(X_train_ovo, Y_train_ovo, True)

            Y_pred_train_ovo = perceptron.predict(X_train_ovo, perceptron.pocket_weights)
            Y_pred_test_ovo = perceptron.predict(X_test_ovo, perceptron.pocket_weights)

            ovo_predictions_train[(class_1, class_2)] = Y_pred_train_ovo
            ovo_predictions_test[(class_1, class_2)] = Y_pred_test_ovo

            if num_features == 2:
                PocketPerceptron.plot_decision_boundary(perceptron, X_train_ovo, Y_train_ovo)

        for (class_1, class_2), pred_train in ovo_predictions_train.items():
            _, _, train_indices = one_vs_one(X_train, Y_train, class_1, class_2)
            for idx, prediction in zip(train_indices, pred_train):
                if prediction == 1:
                    final_votes_train[idx, class_1] += 1
                else:
                    final_votes_train[idx, class_2] += 1

        for (class_1, class_2), pred_test in ovo_predictions_test.items():
            _, _, test_indices = one_vs_one(X_test, Y_test, class_1, class_2)
            for idx, prediction in zip(test_indices, pred_test):
                if prediction == 1:
                    final_votes_test[idx, class_1] += 1
                else:
                    final_votes_test[idx, class_2] += 1

        Y_pred_final_train = np.argmax(final_votes_train, axis=1)
        Y_pred_final_test = np.argmax(final_votes_test, axis=1)

    final_conf_matrix_train = Evaluator.compute_confusion_matrix(Y_train, Y_pred_final_train, class_labels=list(range(num_classes)))
    final_conf_matrix_test = Evaluator.compute_confusion_matrix(Y_test, Y_pred_final_test, class_labels=list(range(num_classes)))

    train_accuracy = Evaluator.overall_accuracy(Y_train, Y_pred_final_train)
    test_accuracy = Evaluator.overall_accuracy(Y_test, Y_pred_final_test)
    train_class_accuracy = Evaluator.class_accuracy(final_conf_matrix_train)
    test_class_accuracy = Evaluator.class_accuracy(final_conf_matrix_test)
    train_producer_acc = Evaluator.producer_accuracy(final_conf_matrix_train)
    test_producer_acc = Evaluator.producer_accuracy(final_conf_matrix_test)
    train_user_acc = Evaluator.user_accuracy(final_conf_matrix_train)
    test_user_acc = Evaluator.user_accuracy(final_conf_matrix_test)
    train_kappa = Evaluator.kappa_coefficient(final_conf_matrix_train)
    test_kappa = Evaluator.kappa_coefficient(final_conf_matrix_test)

    print("\n **Final Evaluation Results**")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Testing Accuracy: {test_accuracy:.4f}")
    print(f"  Training Class Accuracy: {train_class_accuracy}")
    print(f"  Testing Class Accuracy: {test_class_accuracy}")
    print(f"  Training Producer’s Accuracy (Recall): {train_producer_acc}")
    print(f"  Testing Producer’s Accuracy (Recall): {test_producer_acc}")
    print(f"  Training User’s Accuracy (Precision): {train_user_acc}")
    print(f"  Testing User’s Accuracy (Precision): {test_user_acc}")
    print(f"  Training Kappa Coefficient: {train_kappa:.4f}")
    print(f"  Testing Kappa Coefficient: {test_kappa:.4f}")

    Evaluator.plot_confusion_matrix(final_conf_matrix_train, class_labels=list(range(num_classes)), title="Final Confusion Matrix - Train")
    Evaluator.plot_confusion_matrix(final_conf_matrix_test, class_labels=list(range(num_classes)), title="Final Confusion Matrix - Test")

    print("\n **Training and Evaluation Complete for All Digits!**")


# ========== Run Experiments ==========
if __name__ == "__main__":
    run_experiment(feature_set=1, mode="one_vs_all")
    run_experiment(feature_set=1, mode="one_vs_one")
    run_experiment(feature_set=2, mode="one_vs_all")
    run_experiment(feature_set=2, mode="one_vs_one")
