import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "fusion_results"
)

GRAPH_DIR = os.path.join(
    RESULTS_DIR,
    "graphs"
)

os.makedirs(GRAPH_DIR, exist_ok=True)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ================================================================
# LOAD FEDERATED HISTORY
# ================================================================

print("=" * 70)
print("LOADING FEDERATED TRAINING HISTORY")
print("=" * 70)

history_path = os.path.join(
    RESULTS_DIR,
    "privacy_federated_history.json"
)

with open(
    history_path,
    "r",
    encoding="utf-8"
) as file:
    history = json.load(file)

rounds = history["rounds"]
global_accuracy = history["global_accuracy"]
global_loss = history["global_loss"]

print("\nFederated rounds:")
print(rounds)

print("\nGlobal accuracy:")
print(
    [
        round(value * 100, 2)
        for value in global_accuracy
    ]
)

print("\nGlobal loss:")
print(
    [
        round(value, 4)
        for value in global_loss
    ]
)


# ================================================================
# 1. FEDERATED ACCURACY CONVERGENCE
# ================================================================

print("\n" + "=" * 70)
print("CREATING FEDERATED ACCURACY GRAPH")
print("=" * 70)

plt.figure(figsize=(8, 5))

plt.plot(
    rounds,
    [value * 100 for value in global_accuracy],
    marker="o",
    linewidth=2
)

plt.title(
    "Privacy-Aware Federated Learning Convergence"
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Global Test Accuracy (%)"
)

plt.xticks(rounds)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

accuracy_graph_path = os.path.join(
    GRAPH_DIR,
    "federated_accuracy_convergence.png"
)

plt.savefig(
    accuracy_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {accuracy_graph_path}"
)


# ================================================================
# 2. FEDERATED LOSS CONVERGENCE
# ================================================================

print("\n" + "=" * 70)
print("CREATING FEDERATED LOSS GRAPH")
print("=" * 70)

plt.figure(figsize=(8, 5))

plt.plot(
    rounds,
    global_loss,
    marker="o",
    linewidth=2
)

plt.title(
    "Privacy-Aware Federated Learning Loss"
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Global Test Loss"
)

plt.xticks(rounds)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

loss_graph_path = os.path.join(
    GRAPH_DIR,
    "federated_loss_convergence.png"
)

plt.savefig(
    loss_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {loss_graph_path}"
)


# ================================================================
# 3. MODEL ACCURACY COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("CREATING MODEL COMPARISON GRAPH")
print("=" * 70)

model_names = [
    "CNN Baseline",
    "ResNet50",
    "Optimized Fusion",
    "Privacy-Aware\nFederated Fusion"
]

model_accuracies = [
    87.56,
    91.13,
    88.88,
    83.44
]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    model_names,
    model_accuracies
)

plt.title(
    "Brain Tumor Classification Model Comparison"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, accuracy in zip(
    bars,
    model_accuracies
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.tight_layout()

comparison_graph_path = os.path.join(
    GRAPH_DIR,
    "model_accuracy_comparison.png"
)

plt.savefig(
    comparison_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {comparison_graph_path}"
)


# ================================================================
# 4. FINAL FEDERATED CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CREATING FEDERATED CONFUSION MATRIX")
print("=" * 70)

evaluation_path = os.path.join(
    RESULTS_DIR,
    "privacy_federated_test_evaluation.txt"
)

# The confusion matrix is known from the final evaluation.
cm = np.array([
    [250, 103, 34, 13],
    [16, 319, 38, 27],
    [0, 6, 394, 0],
    [7, 21, 0, 372]
])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

disp.plot(
    ax=ax,
    values_format="d",
    cmap="Blues"
)

plt.title(
    "Final Privacy-Aware Federated Model\nConfusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.tight_layout()

confusion_graph_path = os.path.join(
    GRAPH_DIR,
    "federated_confusion_matrix.png"
)

plt.savefig(
    confusion_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {confusion_graph_path}"
)


# ================================================================
# 5. NON-IID CLIENT DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("CREATING NON-IID CLIENT DISTRIBUTION GRAPH")
print("=" * 70)

client_distributions = np.array([
    [800, 400, 100, 100],
    [300, 700, 300, 700],
    [300, 300, 1000, 600]
])

x = np.arange(
    len(CLASS_NAMES)
)

width = 0.25

plt.figure(figsize=(10, 6))

for client_id in range(3):

    plt.bar(
        x + (client_id - 1) * width,
        client_distributions[client_id],
        width,
        label=f"Client {client_id + 1}"
    )

plt.title(
    "Non-IID Training Data Distribution Across Clients"
)

plt.xlabel(
    "Tumor Class"
)

plt.ylabel(
    "Number of Training Images"
)

plt.xticks(
    x,
    CLASS_NAMES
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

client_graph_path = os.path.join(
    GRAPH_DIR,
    "non_iid_client_distribution.png"
)

plt.savefig(
    client_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {client_graph_path}"
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("ALL RESEARCH GRAPHS GENERATED")
print("=" * 70)

print("\nGraphs saved in:")
print(GRAPH_DIR)

print("\nGenerated files:")

print("1. federated_accuracy_convergence.png")
print("2. federated_loss_convergence.png")
print("3. model_accuracy_comparison.png")
print("4. federated_confusion_matrix.png")
print("5. non_iid_client_distribution.png")

print("\nGraph generation completed successfully.")