import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "fusion_results",
    "test_features_pca.npz"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "privacy_federated_fusion_final.keras"
)


# ================================================================
# LOAD TEST DATA
# ================================================================

print("\nLoading test features...")

test_data = np.load(
    FEATURE_FILE
)

test_intensity = test_data["intensity"]
test_texture = test_data["texture"]
test_spatial = test_data["spatial"]
test_deep = test_data["deep"]
test_labels = test_data["labels"]


print(
    f"Test samples: {len(test_labels)}"
)


# ================================================================
# LOAD FEDERATED MODEL
# ================================================================

print("\nLoading federated model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    safe_mode=False
)

print("Federated model loaded successfully.")


# ================================================================
# PREDICTIONS
# ================================================================

print("\nGenerating predictions...")

x_test = [
    test_intensity,
    test_texture,
    test_spatial,
    test_deep
]

probabilities = model.predict(
    x_test,
    batch_size=32,
    verbose=1
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)


# ================================================================
# METRICS
# ================================================================

accuracy = accuracy_score(
    test_labels,
    predicted_labels
)

precision = precision_score(
    test_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    test_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    test_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# ================================================================
# DISPLAY RESULTS
# ================================================================

print("\n")
print("=" * 70)
print("PRIVACY-AWARE FEDERATED MODEL EVALUATION")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-score  : {f1 * 100:.2f}%"
)


# ================================================================
# CLASS-WISE REPORT
# ================================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

print("\n")
print("=" * 70)
print("CLASS-WISE CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

matrix = confusion_matrix(
    test_labels,
    predicted_labels
)

# ================================================================
# CLASS-WISE SPECIFICITY
# ================================================================

specificities = []

for class_index in range(
    len(class_names)
):

    true_negative = (
        matrix.sum()
        - matrix[class_index, :].sum()
        - matrix[:, class_index].sum()
        + matrix[class_index, class_index]
    )

    false_positive = (
        matrix[:, class_index].sum()
        - matrix[class_index, class_index]
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0.0
    )

    specificities.append(
        specificity
    )

macro_specificity = np.mean(
    specificities
)
# ================================================================
# ROC-AUC
# ================================================================

roc_auc = roc_auc_score(
    test_labels,
    probabilities,
    multi_class="ovr",
    average=None
)

macro_roc_auc = roc_auc_score(
    test_labels,
    probabilities,
    multi_class="ovr",
    average="macro"
)

print("\n")

print("=" * 70)

print("CONFUSION MATRIX")

print("=" * 70)

print(matrix)

print("\n")
print("=" * 70)
print("SPECIFICITY")
print("=" * 70)

for class_name, specificity in zip(
    class_names,
    specificities
):
    print(
        f"{class_name:<15}: "
        f"{specificity * 100:.2f}%"
    )

print(
    f"\nMacro Specificity: "
    f"{macro_specificity * 100:.2f}%"
)

print("\n")
print("=" * 70)
print("ROC-AUC")
print("=" * 70)

for class_name, auc_value in zip(
    class_names,
    roc_auc
):
    print(
        f"{class_name:<15}: "
        f"{auc_value:.4f}"
    )

print(
    f"\nMacro ROC-AUC: "
    f"{macro_roc_auc:.4f}"
)


# ================================================================
# SAVE EVALUATION RESULTS
# ================================================================

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "fusion_results",
    "federated_evaluation_metrics.txt"
)

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PRIVACY-AWARE FEDERATED MODEL EVALUATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Test Samples: {len(test_labels)}\n\n"
    )

    file.write(
        "Overall Evaluation Metrics\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Accuracy  : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision : {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall    : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1-score  : {f1 * 100:.2f}%\n"
    )

    file.write("\n")

    file.write(
        "Class-wise Classification Report\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        classification_report(
            test_labels,
            predicted_labels,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    file.write("\n")

    file.write(
        "Confusion Matrix\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        np.array2string(matrix)
    )

    file.write("\n\n")

    file.write(
        "Class-wise Specificity\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for class_name, specificity in zip(
        class_names,
        specificities
    ):

        file.write(
            f"{class_name:<15}: "
            f"{specificity * 100:.2f}%\n"
        )

    file.write(
        f"\nMacro Specificity: "
        f"{macro_specificity * 100:.2f}%\n"
    )

    file.write("\n")

    file.write(
        "Class-wise ROC-AUC\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for class_name, auc_value in zip(
        class_names,
        roc_auc
    ):

        file.write(
            f"{class_name:<15}: "
            f"{auc_value:.4f}\n"
        )

    file.write(
        f"\nMacro ROC-AUC: "
        f"{macro_roc_auc:.4f}\n"
    )

print(
    f"\nEvaluation results saved to:\n{RESULTS_FILE}"
)