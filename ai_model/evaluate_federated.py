import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FUSION_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "fusion_results"
)

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "privacy_federated_fusion_final.keras"
)


# ================================================================
# LOAD TEST FEATURES
# ================================================================

print("=" * 70)
print("LOADING TEST FEATURES")
print("=" * 70)

test_data = np.load(
    os.path.join(
        FUSION_RESULTS_DIR,
        "test_features_pca.npz"
    )
)

test_intensity = test_data["intensity"]
test_texture = test_data["texture"]
test_spatial = test_data["spatial"]
test_deep = test_data["deep"]
test_labels = test_data["labels"]

print(f"Intensity : {test_intensity.shape}")
print(f"Texture   : {test_texture.shape}")
print(f"Spatial   : {test_spatial.shape}")
print(f"Deep      : {test_deep.shape}")
print(f"Labels    : {test_labels.shape}")


# ================================================================
# RECREATE THE SAME OPTIMIZED FUSION ARCHITECTURE
# ================================================================

print("\n" + "=" * 70)
print("REBUILDING OPTIMIZED FUSION ARCHITECTURE")
print("=" * 70)


def build_fusion_model():

    intensity_input = layers.Input(
        shape=(32,),
        name="intensity_input"
    )

    texture_input = layers.Input(
        shape=(32,),
        name="texture_input"
    )

    spatial_input = layers.Input(
        shape=(32,),
        name="spatial_input"
    )

    deep_input = layers.Input(
        shape=(128,),
        name="deep_input"
    )

    intensity_branch = layers.Dense(
        32,
        activation="relu",
        name="intensity_branch"
    )(intensity_input)

    texture_branch = layers.Dense(
        32,
        activation="relu",
        name="texture_branch"
    )(texture_input)

    spatial_branch = layers.Dense(
        32,
        activation="relu",
        name="spatial_branch"
    )(spatial_input)

    deep_branch = layers.Dense(
        64,
        activation="relu",
        name="deep_branch"
    )(deep_input)

    combined_features = layers.Concatenate(
        name="combined_branch_features"
    )([
        intensity_branch,
        texture_branch,
        spatial_branch,
        deep_branch
    ])

    gate_hidden = layers.Dense(
        64,
        activation="relu",
        name="gate_hidden"
    )(combined_features)

    gate_weights = layers.Dense(
        4,
        activation="softmax",
        name="feature_gate_weights"
    )(gate_hidden)

    intensity_gate = layers.Lambda(
        lambda x: x[:, 0:1],
        name="intensity_gate"
    )(gate_weights)

    texture_gate = layers.Lambda(
        lambda x: x[:, 1:2],
        name="texture_gate"
    )(gate_weights)

    spatial_gate = layers.Lambda(
        lambda x: x[:, 2:3],
        name="spatial_gate"
    )(gate_weights)

    deep_gate = layers.Lambda(
        lambda x: x[:, 3:4],
        name="deep_gate"
    )(gate_weights)

    weighted_intensity = layers.Multiply(
        name="weighted_intensity"
    )([
        intensity_branch,
        intensity_gate
    ])

    weighted_texture = layers.Multiply(
        name="weighted_texture"
    )([
        texture_branch,
        texture_gate
    ])

    weighted_spatial = layers.Multiply(
        name="weighted_spatial"
    )([
        spatial_branch,
        spatial_gate
    ])

    weighted_deep = layers.Multiply(
        name="weighted_deep"
    )([
        deep_branch,
        deep_gate
    ])

    fused_features = layers.Concatenate(
        name="optimized_fused_features"
    )([
        weighted_intensity,
        weighted_texture,
        weighted_spatial,
        weighted_deep
    ])

    fused_representation = layers.Dense(
        128,
        activation="relu",
        name="fused_representation"
    )(fused_features)

    fused_representation = layers.Dropout(
        0.30,
        name="fusion_dropout"
    )(fused_representation)

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classifier"
    )(fused_representation)

    model = models.Model(
        inputs=[
            intensity_input,
            texture_input,
            spatial_input,
            deep_input
        ],
        outputs=output,
        name="privacy_aware_optimized_fusion"
    )

    return model


model = build_fusion_model()

print("Architecture rebuilt successfully.")


# ================================================================
# LOAD FINAL FEDERATED MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING FINAL FEDERATED MODEL")
print("=" * 70)

trained_model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False
)

print("Final federated model loaded successfully.")


# ================================================================
# COPY TRAINED WEIGHTS
# ================================================================

model.set_weights(
    trained_model.get_weights()
)

print("Federated weights transferred successfully.")


# ================================================================
# PREDICTION
# ================================================================

print("\n" + "=" * 70)
print("EVALUATING FINAL FEDERATED MODEL")
print("=" * 70)

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

print("\n" + "=" * 70)
print("FINAL FEDERATED TEST RESULTS")
print("=" * 70)

print(
    f"\nTest Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision   : {precision * 100:.2f}%"
)

print(
    f"Weighted Recall      : {recall * 100:.2f}%"
)

print(
    f"Weighted F1-Score    : {f1 * 100:.2f}%"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    test_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(report)


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    test_labels,
    predicted_labels
)

print("-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print(cm)


# ================================================================
# SAVE RESULTS
# ================================================================

RESULTS_PATH = os.path.join(
    FUSION_RESULTS_DIR,
    "privacy_federated_test_evaluation.txt"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FINAL PRIVACY-AWARE FEDERATED TEST EVALUATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Test Accuracy        : "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Precision   : "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Recall      : "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"Weighted F1-Score    : "
        f"{f1 * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write("-" * 70 + "\n")

    file.write(report)

    file.write("\n")
    file.write(
        "CONFUSION MATRIX\n"
    )

    file.write("-" * 70 + "\n")

    file.write(
        np.array2string(cm)
    )

print("\n" + "=" * 70)
print("FEDERATED EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(RESULTS_PATH)