import os
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "fusion_results",
    "graphs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# FINAL EVALUATION RESULTS
# ================================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

accuracy = 83.44
precision = 84.37
recall = 83.44
f1 = 83.02
specificity = 94.48

class_precision = [
    91.58,
    71.05,
    84.55,
    90.29
]

class_recall = [
    62.50,
    79.75,
    98.50,
    93.00
]

class_f1 = [
    74.29,
    75.15,
    90.99,
    91.63
]

class_specificity = [
    98.08,
    89.17,
    94.00,
    96.67
]

roc_auc = [
    0.9224,
    0.9321,
    0.9968,
    0.9916
]


# ================================================================
# GRAPH 1 — OVERALL METRICS
# ================================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Specificity"
]

values = [
    accuracy,
    precision,
    recall,
    f1,
    specificity
]

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    metrics,
    values
)

plt.title(
    "Overall Evaluation Metrics"
)

plt.ylabel(
    "Score (%)"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "overall_evaluation_metrics.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# GRAPH 2 — CLASS-WISE PERFORMANCE
# ================================================================

x = np.arange(
    len(class_names)
)

width = 0.2

plt.figure(
    figsize=(12, 7)
)

plt.bar(
    x - 1.5 * width,
    class_precision,
    width,
    label="Precision"
)

plt.bar(
    x - 0.5 * width,
    class_recall,
    width,
    label="Recall"
)

plt.bar(
    x + 0.5 * width,
    class_f1,
    width,
    label="F1-score"
)

plt.bar(
    x + 1.5 * width,
    class_specificity,
    width,
    label="Specificity"
)

plt.xticks(
    x,
    class_names
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Class-wise Evaluation Performance"
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_wise_evaluation_metrics.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# GRAPH 3 — ROC-AUC
# ================================================================

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    class_names,
    roc_auc
)

plt.title(
    "Class-wise ROC-AUC"
)

plt.ylabel(
    "ROC-AUC"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    roc_auc
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.4f}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_wise_roc_auc.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# GRAPH 4 — CONFUSION MATRIX
# ================================================================

confusion_matrix = np.array([
    [250, 103, 34, 13],
    [16, 319, 38, 27],
    [0, 6, 394, 0],
    [7, 21, 0, 372]
])

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title(
    "Federated Model Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=30,
    ha="right"
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

threshold = confusion_matrix.max() / 2

for i in range(
    confusion_matrix.shape[0]
):

    for j in range(
        confusion_matrix.shape[1]
    ):

        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center",
            color="white"
            if confusion_matrix[i, j] > threshold
            else "black",
            fontweight="bold"
        )

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "federated_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# COMPLETE
# ================================================================

print("\nEvaluation graphs generated successfully.")

print(
    f"\nSaved to:\n{OUTPUT_DIR}"
)

print("\nGenerated files:")

print(
    "1. overall_evaluation_metrics.png"
)

print(
    "2. class_wise_evaluation_metrics.png"
)

print(
    "3. class_wise_roc_auc.png"
)

print(
    "4. federated_confusion_matrix.png"
)