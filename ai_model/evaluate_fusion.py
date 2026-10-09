import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\admin\Desktop\BrainTumorAI"
AI_MODEL_DIR = os.path.join(BASE_DIR, "ai_model")

RESULTS_DIR = os.path.join(
    AI_MODEL_DIR,
    "fusion_results"
)

MODEL_DIR = os.path.join(
    AI_MODEL_DIR,
    "models"
)

PCA_TRAIN_CACHE = os.path.join(
    RESULTS_DIR,
    "train_features_pca.npz"
)

PCA_TEST_CACHE = os.path.join(
    RESULTS_DIR,
    "test_features_pca.npz"
)

FUSION_BEST_MODEL = os.path.join(
    MODEL_DIR,
    "optimized_fusion_best.keras"
)

BATCH_SIZE = 32

NUM_CLASSES = 4

INTENSITY_FEATURE_DIM = 32
TEXTURE_FEATURE_DIM = 32
SPATIAL_FEATURE_DIM = 32
DEEP_FEATURE_DIM = 128
FUSED_FEATURE_DIM = 128

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# LOAD PCA-REDUCED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("LOADING PCA-REDUCED FEATURES")
print("=" * 70)

train_data = np.load(PCA_TRAIN_CACHE)
test_data = np.load(PCA_TEST_CACHE)

train_intensity = train_data["intensity"].astype(np.float32)
train_texture = train_data["texture"].astype(np.float32)
train_spatial = train_data["spatial"].astype(np.float32)
train_deep = train_data["deep"].astype(np.float32)
train_labels = train_data["labels"].astype(np.int32)

test_intensity = test_data["intensity"].astype(np.float32)
test_texture = test_data["texture"].astype(np.float32)
test_spatial = test_data["spatial"].astype(np.float32)
test_deep = test_data["deep"].astype(np.float32)
test_labels = test_data["labels"].astype(np.int32)

print("\nTraining features:")
print("Intensity:", train_intensity.shape)
print("Texture :", train_texture.shape)
print("Spatial :", train_spatial.shape)
print("Deep    :", train_deep.shape)

print("\nTesting features:")
print("Intensity:", test_intensity.shape)
print("Texture :", test_texture.shape)
print("Spatial :", test_spatial.shape)
print("Deep    :", test_deep.shape)


# ============================================================
# RECREATE TRAINING STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("RECREATING FEATURE STANDARDIZATION")
print("=" * 70)

intensity_scaler = StandardScaler()
texture_scaler = StandardScaler()
spatial_scaler = StandardScaler()
deep_scaler = StandardScaler()

train_intensity_scaled = intensity_scaler.fit_transform(
    train_intensity
).astype(np.float32)

test_intensity_scaled = intensity_scaler.transform(
    test_intensity
).astype(np.float32)


train_texture_scaled = texture_scaler.fit_transform(
    train_texture
).astype(np.float32)

test_texture_scaled = texture_scaler.transform(
    test_texture
).astype(np.float32)


train_spatial_scaled = spatial_scaler.fit_transform(
    train_spatial
).astype(np.float32)

test_spatial_scaled = spatial_scaler.transform(
    test_spatial
).astype(np.float32)


train_deep_scaled = deep_scaler.fit_transform(
    train_deep
).astype(np.float32)

test_deep_scaled = deep_scaler.transform(
    test_deep
).astype(np.float32)

print("\nFeature standardization recreated successfully.")


# ============================================================
# REBUILD IDENTICAL FUSION ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("REBUILDING OPTIMIZED FUSION ARCHITECTURE")
print("=" * 70)


def build_optimized_fusion_model():

    intensity_input = layers.Input(
        shape=(INTENSITY_FEATURE_DIM,),
        name="intensity_input"
    )

    texture_input = layers.Input(
        shape=(TEXTURE_FEATURE_DIM,),
        name="texture_input"
    )

    spatial_input = layers.Input(
        shape=(SPATIAL_FEATURE_DIM,),
        name="spatial_input"
    )

    deep_input = layers.Input(
        shape=(DEEP_FEATURE_DIM,),
        name="deep_input"
    )


    intensity_branch = layers.Dense(
        32,
        activation="relu",
        name="intensity_projection"
    )(intensity_input)


    texture_branch = layers.Dense(
        32,
        activation="relu",
        name="texture_projection"
    )(texture_input)


    spatial_branch = layers.Dense(
        32,
        activation="relu",
        name="spatial_projection"
    )(spatial_input)


    deep_branch = layers.Dense(
        64,
        activation="relu",
        name="deep_projection"
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
        name="fusion_gate_hidden"
    )(combined_features)


    gate_weights = layers.Dense(
        4,
        activation="softmax",
        name="fusion_gate_weights"
    )(gate_hidden)


    intensity_weight = layers.Lambda(
        lambda x: x[:, 0:1],
        name="intensity_gate"
    )(gate_weights)


    texture_weight = layers.Lambda(
        lambda x: x[:, 1:2],
        name="texture_gate"
    )(gate_weights)


    spatial_weight = layers.Lambda(
        lambda x: x[:, 2:3],
        name="spatial_gate"
    )(gate_weights)


    deep_weight = layers.Lambda(
        lambda x: x[:, 3:4],
        name="deep_gate"
    )(gate_weights)


    weighted_intensity = layers.Multiply(
        name="weighted_intensity"
    )([
        intensity_branch,
        intensity_weight
    ])


    weighted_texture = layers.Multiply(
        name="weighted_texture"
    )([
        texture_branch,
        texture_weight
    ])


    weighted_spatial = layers.Multiply(
        name="weighted_spatial"
    )([
        spatial_branch,
        spatial_weight
    ])


    weighted_deep = layers.Multiply(
        name="weighted_deep"
    )([
        deep_branch,
        deep_weight
    ])


    fused_features = layers.Concatenate(
        name="optimized_fused_features"
    )([
        weighted_intensity,
        weighted_texture,
        weighted_spatial,
        weighted_deep
    ])


    fused_features = layers.Dense(
        FUSED_FEATURE_DIM,
        activation="relu",
        name="fused_feature_projection"
    )(fused_features)


    fused_features = layers.Dropout(
        0.30,
        name="fusion_dropout"
    )(fused_features)


    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="tumor_classifier"
    )(fused_features)


    model = models.Model(
        inputs=[
            intensity_input,
            texture_input,
            spatial_input,
            deep_input
        ],
        outputs=output,
        name="optimized_gated_feature_fusion"
    )

    return model


fusion_model = build_optimized_fusion_model()

print("\nFusion architecture rebuilt successfully.")


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED FUSION WEIGHTS")
print("=" * 70)

saved_model = tf.keras.models.load_model(
    FUSION_BEST_MODEL,
    safe_mode=False
)

fusion_model.set_weights(
    saved_model.get_weights()
)

print("\nTrained fusion weights loaded successfully.")


# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON 1600-IMAGE TEST SET")
print("=" * 70)

test_probabilities = fusion_model.predict(
    [
        test_intensity_scaled,
        test_texture_scaled,
        test_spatial_scaled,
        test_deep_scaled
    ],
    batch_size=BATCH_SIZE,
    verbose=1
)


test_predictions = np.argmax(
    test_probabilities,
    axis=1
)


# ============================================================
# CALCULATE METRICS
# ============================================================

test_accuracy = np.mean(
    test_predictions == test_labels
)

precision = precision_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print("\n" + "=" * 70)
print("CENTRALIZED OPTIMIZED FUSION TEST RESULTS")
print("=" * 70)

print(
    f"\nTest Accuracy  : {test_accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision : {precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : {recall * 100:.2f}%"
)

print(
    f"Weighted F1-Score  : {f1 * 100:.2f}%"
)


print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

evaluation_path = os.path.join(
    RESULTS_DIR,
    "optimized_fusion_test_evaluation.txt"
)

with open(evaluation_path, "w") as f:

    f.write(
        "CENTRALIZED OPTIMIZED FEATURE-FUSION MODEL\n"
    )

    f.write(
        "============================================\n\n"
    )

    f.write(
        f"Test Accuracy: {test_accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Weighted Precision: {precision * 100:.4f}%\n"
    )

    f.write(
        f"Weighted Recall: {recall * 100:.4f}%\n"
    )

    f.write(
        f"Weighted F1-Score: {f1 * 100:.4f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "---------------------\n"
    )

    f.write(report)

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        "----------------\n"
    )

    f.write(str(cm))


print("\n" + "=" * 70)
print("FUSION TEST EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved to:")

print(evaluation_path)