import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ================================================================
# PRIVACY-AWARE FEDERATED OPTIMIZED FUSION TRAINING
# STEP 1: CONFIGURATION AND DATA LOADING
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FUSION_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "fusion_results"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(FUSION_RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------
# Federated learning configuration
# ------------------------------------------------

NUM_CLIENTS = 3
FEDERATED_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ------------------------------------------------
# Privacy configuration
# ------------------------------------------------
#
# CLIP_NORM:
# Maximum allowed L2 norm of a client's model update.
#
# NOISE_MULTIPLIER:
# Gaussian noise scale applied after clipping.
#
# This implementation will be described as
# DP-style update perturbation unless a formal
# privacy accountant is added.
# ------------------------------------------------

CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.01


# ------------------------------------------------
# Learning configuration
# ------------------------------------------------

LEARNING_RATE = 0.001


print("=" * 70)
print("PRIVACY-AWARE FEDERATED OPTIMIZED FUSION")
print("=" * 70)

print("\nConfiguration")
print("-" * 70)
print(f"Number of clients       : {NUM_CLIENTS}")
print(f"Federated rounds        : {FEDERATED_ROUNDS}")
print(f"Local epochs            : {LOCAL_EPOCHS}")
print(f"Batch size              : {BATCH_SIZE}")
print(f"Learning rate           : {LEARNING_RATE}")
print(f"Clipping norm           : {CLIP_NORM}")
print(f"Noise multiplier        : {NOISE_MULTIPLIER}")


# ================================================================
# LOAD PCA-REDUCED FEATURES
# ================================================================

print("\n" + "=" * 70)
print("LOADING PCA-REDUCED FEATURES")
print("=" * 70)

train_data = np.load(
    os.path.join(
        FUSION_RESULTS_DIR,
        "train_features_pca.npz"
    )
)

test_data = np.load(
    os.path.join(
        FUSION_RESULTS_DIR,
        "test_features_pca.npz"
    )
)


# Training features
train_intensity = train_data["intensity"]
train_texture = train_data["texture"]
train_spatial = train_data["spatial"]
train_deep = train_data["deep"]
train_labels = train_data["labels"]


# Testing features
test_intensity = test_data["intensity"]
test_texture = test_data["texture"]
test_spatial = test_data["spatial"]
test_deep = test_data["deep"]
test_labels = test_data["labels"]


print("\nTraining features:")
print(f"Intensity : {train_intensity.shape}")
print(f"Texture   : {train_texture.shape}")
print(f"Spatial   : {train_spatial.shape}")
print(f"Deep      : {train_deep.shape}")

print("\nTesting features:")
print(f"Intensity : {test_intensity.shape}")
print(f"Texture   : {test_texture.shape}")
print(f"Spatial   : {test_spatial.shape}")
print(f"Deep      : {test_deep.shape}")


# ================================================================
# LOAD NON-IID CLIENT INDICES
# ================================================================

print("\n" + "=" * 70)
print("LOADING NON-IID CLIENT PARTITIONS")
print("=" * 70)

client_index_data = np.load(
    os.path.join(
        FUSION_RESULTS_DIR,
        "non_iid_client_indices.npz"
    )
)

client_indices = [
    client_index_data["client_1"],
    client_index_data["client_2"],
    client_index_data["client_3"]
]


# ------------------------------------------------
# Verify client partitions
# ------------------------------------------------

all_client_indices = np.concatenate(client_indices)

assert len(all_client_indices) == len(train_labels)

assert len(np.unique(all_client_indices)) == len(train_labels)

assert set(all_client_indices.tolist()) == set(
    range(len(train_labels))
)


print("Client partition verification PASSED.")

for client_id, indices in enumerate(client_indices):

    print(
        f"Client {client_id + 1}: "
        f"{len(indices)} training samples"
    )


print("\n" + "=" * 70)
print("DATA LOADING COMPLETED SUCCESSFULLY")
print("=" * 70)
# ================================================================
# OPTIMIZED GATED FEATURE-FUSION MODEL
# ================================================================

from tensorflow.keras import layers, models


def build_fusion_model(learning_rate=0.001):

    # ------------------------------------------------------------
    # Four feature inputs
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Feature branch projections
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Concatenate branch representations
    # ------------------------------------------------------------

    combined_features = layers.Concatenate(
        name="combined_branch_features"
    )([
        intensity_branch,
        texture_branch,
        spatial_branch,
        deep_branch
    ])

    # ------------------------------------------------------------
    # Learnable feature gates
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Extract one gate value for each feature branch
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Apply learned gates
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Final feature fusion
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Classification layer
    # ------------------------------------------------------------

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classifier"
    )(fused_representation)

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ================================================================
# TEST MODEL ARCHITECTURE
# ================================================================

print("\n" + "=" * 70)
print("BUILDING OPTIMIZED GATED FUSION MODEL")
print("=" * 70)

test_model = build_fusion_model(
    learning_rate=LEARNING_RATE
)

print("\nModel created successfully.")
print(f"Total parameters     : {test_model.count_params():,}")
print(
    f"Trainable parameters : "
    f"{sum(np.prod(v.shape) for v in test_model.trainable_weights):,}"
)

print("\nOptimized gated fusion architecture is ready.")
# ================================================================
# PRIVACY-AWARE CLIENT UPDATE PROCESSING
# ================================================================

def calculate_update_norm(client_weights, global_weights):
    """
    Calculate the L2 norm of the difference between
    client model weights and global model weights.
    """

    squared_sum = 0.0

    for client_weight, global_weight in zip(
        client_weights,
        global_weights
    ):
        difference = client_weight - global_weight
        squared_sum += np.sum(
            np.square(difference)
        )

    return float(np.sqrt(squared_sum))


def clip_model_update(
    client_weights,
    global_weights,
    clip_norm
):
    """
    Clip the complete client model update to the
    specified L2 norm.
    """

    # Calculate update norm
    update_norm = calculate_update_norm(
        client_weights,
        global_weights
    )

    # Calculate clipping factor
    if update_norm > clip_norm:
        clipping_factor = clip_norm / (
            update_norm + 1e-12
        )
    else:
        clipping_factor = 1.0

    clipped_weights = []

    for client_weight, global_weight in zip(
        client_weights,
        global_weights
    ):

        update = client_weight - global_weight

        clipped_update = (
            update * clipping_factor
        )

        clipped_weight = (
            global_weight + clipped_update
        )

        clipped_weights.append(
            clipped_weight
        )

    clipped_norm = calculate_update_norm(
        clipped_weights,
        global_weights
    )

    return (
        clipped_weights,
        update_norm,
        clipped_norm,
        clipping_factor
    )


def add_gaussian_noise(
    clipped_weights,
    global_weights,
    noise_multiplier,
    clip_norm
):
    """
    Add Gaussian noise to the clipped client update.
    """

    noisy_weights = []

    noise_std = (
        noise_multiplier * clip_norm
    )

    total_noise_squared = 0.0
    total_parameters = 0

    for clipped_weight, global_weight in zip(
        clipped_weights,
        global_weights
    ):

        clipped_update = (
            clipped_weight - global_weight
        )

        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=clipped_update.shape
        ).astype(
            clipped_update.dtype
        )

        noisy_update = (
            clipped_update + noise
        )

        noisy_weight = (
            global_weight + noisy_update
        )

        noisy_weights.append(
            noisy_weight
        )

        total_noise_squared += np.sum(
            np.square(noise)
        )

        total_parameters += noise.size

    noise_l2_norm = float(
        np.sqrt(total_noise_squared)
    )

    average_noise_magnitude = float(
        np.sqrt(
            total_noise_squared /
            max(total_parameters, 1)
        )
    )

    return (
        noisy_weights,
        noise_l2_norm,
        average_noise_magnitude
    )


# ================================================================
# TEST PRIVACY OPERATIONS
# ================================================================

print("\n" + "=" * 70)
print("TESTING PRIVACY UPDATE OPERATIONS")
print("=" * 70)

privacy_test_model = build_fusion_model(
    learning_rate=LEARNING_RATE
)

privacy_global_weights = [
    weight.copy()
    for weight in privacy_test_model.get_weights()
]

# Create a simulated client update
privacy_client_weights = [
    weight.copy()
    for weight in privacy_global_weights
]

# Add a small artificial change
for i in range(len(privacy_client_weights)):
    privacy_client_weights[i] = (
        privacy_client_weights[i]
        + np.random.normal(
            0,
            0.01,
            size=privacy_client_weights[i].shape
        ).astype(
            privacy_client_weights[i].dtype
        )
    )


# ------------------------------------------------
# Measure original update
# ------------------------------------------------

original_norm = calculate_update_norm(
    privacy_client_weights,
    privacy_global_weights
)

print(f"\nOriginal update norm : {original_norm:.6f}")


# ------------------------------------------------
# Apply clipping
# ------------------------------------------------

(
    clipped_weights,
    before_clip_norm,
    after_clip_norm,
    clipping_factor
) = clip_model_update(
    privacy_client_weights,
    privacy_global_weights,
    CLIP_NORM
)

print(
    f"After clipping norm  : {after_clip_norm:.6f}"
)

print(
    f"Clipping factor      : {clipping_factor:.6f}"
)


# ------------------------------------------------
# Add Gaussian noise
# ------------------------------------------------

(
    noisy_weights,
    noise_l2_norm,
    average_noise_magnitude
) = add_gaussian_noise(
    clipped_weights,
    privacy_global_weights,
    NOISE_MULTIPLIER,
    CLIP_NORM
)

print(
    f"Noise L2 norm        : {noise_l2_norm:.6f}"
)

print(
    f"Average noise value  : "
    f"{average_noise_magnitude:.8f}"
)


# ------------------------------------------------
# Final verification
# ------------------------------------------------

assert len(noisy_weights) == len(
    privacy_global_weights
)

for noisy_weight, global_weight in zip(
    noisy_weights,
    privacy_global_weights
):

    assert noisy_weight.shape == global_weight.shape


print("\nPrivacy update processing PASSED.")
# ================================================================
# FEDERATED LOCAL TRAINING
# ================================================================

def train_client(
    global_weights,
    client_data,
    client_id,
    round_number
):
    """
    Train one client locally starting from the current
    global model weights.
    """

    print(
        f"\nClient {client_id + 1} "
        f"local training..."
    )

    # ------------------------------------------------------------
    # Create a fresh copy of the fusion model
    # ------------------------------------------------------------

    client_model = build_fusion_model(
        learning_rate=LEARNING_RATE
    )

    client_model.set_weights(global_weights)

    # ------------------------------------------------------------
    # Prepare client data
    # ------------------------------------------------------------

    x_client = [
        client_data["intensity"],
        client_data["texture"],
        client_data["spatial"],
        client_data["deep"]
    ]

    y_client = client_data["labels"]

    # ------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------

    history = client_model.fit(
        x=x_client,
        y=y_client,
        batch_size=BATCH_SIZE,
        epochs=LOCAL_EPOCHS,
        verbose=0,
        shuffle=True
    )

    local_accuracy = float(
        history.history["accuracy"][-1]
    )

    local_loss = float(
        history.history["loss"][-1]
    )

    print(
        f"  Local loss     : {local_loss:.4f}"
    )

    print(
        f"  Local accuracy : "
        f"{local_accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Obtain locally trained weights
    # ------------------------------------------------------------

    client_weights = [
        weight.copy()
        for weight in client_model.get_weights()
    ]

    return (
        client_weights,
        len(y_client),
        local_loss,
        local_accuracy
    )


# ================================================================
# PREPARE CLIENT DATA
# ================================================================

print("\n" + "=" * 70)
print("PREPARING FEDERATED CLIENT DATA")
print("=" * 70)

federated_clients = []

for client_id in range(NUM_CLIENTS):

    indices = client_indices[client_id]

    client_data = {
        "intensity": train_intensity[indices],
        "texture": train_texture[indices],
        "spatial": train_spatial[indices],
        "deep": train_deep[indices],
        "labels": train_labels[indices]
    }

    federated_clients.append(client_data)

    print(
        f"Client {client_id + 1}: "
        f"{len(indices)} samples"
    )


# ================================================================
# FEDAVG AGGREGATION
# ================================================================

def federated_average(
    client_weights,
    client_sample_counts
):
    """
    Weighted Federated Averaging (FedAvg).

    Larger clients receive proportionally larger
    aggregation weights.
    """

    total_samples = sum(
        client_sample_counts
    )

    client_weights_normalized = [
        count / total_samples
        for count in client_sample_counts
    ]

    aggregated_weights = []

    for layer_index in range(
        len(client_weights[0])
    ):

        weighted_layer = np.zeros_like(
            client_weights[0][layer_index]
        )

        for client_id in range(NUM_CLIENTS):

            weighted_layer += (
                client_weights[client_id][layer_index]
                * client_weights_normalized[client_id]
            )

        aggregated_weights.append(
            weighted_layer
        )

    return aggregated_weights


# ================================================================
# GLOBAL TEST EVALUATION
# ================================================================

def evaluate_global_model(
    global_weights
):
    """
    Evaluate the current global model on the
    independent 1600-image test set.

    Returns:
        test_loss
        test_accuracy
        precision
        recall
        f1_score
        classification_report
        confusion_matrix
    """

    evaluation_model = build_fusion_model(
        learning_rate=LEARNING_RATE
    )

    evaluation_model.set_weights(
        global_weights
    )

    x_test = [
        test_intensity,
        test_texture,
        test_spatial,
        test_deep
    ]

    test_loss, test_accuracy = (
        evaluation_model.evaluate(
            x_test,
            test_labels,
            batch_size=BATCH_SIZE,
            verbose=0
        )
    )

    # ------------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------------

    prediction_probabilities = (
        evaluation_model.predict(
            x_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )
    )

    predicted_labels = np.argmax(
        prediction_probabilities,
        axis=1
    )

    # ------------------------------------------------------------
    # Calculate evaluation metrics
    # ------------------------------------------------------------

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

    report = classification_report(
        test_labels,
        predicted_labels,
        target_names=[
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary"
        ],
        digits=4,
        zero_division=0
    )

    matrix = confusion_matrix(
        test_labels,
        predicted_labels
    )

    return (
        float(test_loss),
        float(test_accuracy),
        float(precision),
        float(recall),
        float(f1),
        report,
        matrix
    )

# ================================================================
# PRIVACY-AWARE FEDERATED TRAINING
# ================================================================

print("\n" + "=" * 70)
print("STARTING PRIVACY-AWARE FEDERATED TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# Initialize global model
# ------------------------------------------------------------

global_model = build_fusion_model(
    learning_rate=LEARNING_RATE
)

global_weights = [
    weight.copy()
    for weight in global_model.get_weights()
]


# ------------------------------------------------------------
# Store training history
# ------------------------------------------------------------

federated_history = {
    "rounds": [],
    "global_accuracy": [],
    "global_loss": [],
    "client_metrics": [],
    "privacy_metrics": []
}


# ================================================================
# FEDERATED ROUNDS
# ================================================================

for round_number in range(
    1,
    FEDERATED_ROUNDS + 1
):

    print("\n" + "=" * 70)
    print(
        f"FEDERATED ROUND {round_number}/"
        f"{FEDERATED_ROUNDS}"
    )
    print("=" * 70)

    round_client_weights = []
    round_sample_counts = []

    round_client_metrics = []
    round_privacy_metrics = []

    # ------------------------------------------------------------
    # Train each client
    # ------------------------------------------------------------

    for client_id in range(NUM_CLIENTS):

        client_weights, sample_count, local_loss, local_accuracy = (
            train_client(
                global_weights,
                federated_clients[client_id],
                client_id,
                round_number
            )
        )

        # --------------------------------------------------------
        # Measure update before clipping
        # --------------------------------------------------------

        (
            clipped_weights,
            before_clip_norm,
            after_clip_norm,
            clipping_factor
        ) = clip_model_update(
            client_weights,
            global_weights,
            CLIP_NORM
        )

        # --------------------------------------------------------
        # Add Gaussian noise
        # --------------------------------------------------------

        (
            noisy_weights,
            noise_l2_norm,
            average_noise_magnitude
        ) = add_gaussian_noise(
            clipped_weights,
            global_weights,
            NOISE_MULTIPLIER,
            CLIP_NORM
        )

        # --------------------------------------------------------
        # Store processed client update
        # --------------------------------------------------------

        round_client_weights.append(
            noisy_weights
        )

        round_sample_counts.append(
            sample_count
        )

        # --------------------------------------------------------
        # Store metrics
        # --------------------------------------------------------

        round_client_metrics.append({
            "client_id": client_id + 1,
            "samples": sample_count,
            "loss": local_loss,
            "accuracy": local_accuracy
        })

        round_privacy_metrics.append({
            "client_id": client_id + 1,
            "before_clip_norm": before_clip_norm,
            "after_clip_norm": after_clip_norm,
            "clipping_factor": clipping_factor,
            "noise_l2_norm": noise_l2_norm,
            "average_noise_magnitude": average_noise_magnitude
        })

        print(
            f"  Update norm before clipping : "
            f"{before_clip_norm:.6f}"
        )

        print(
            f"  Update norm after clipping  : "
            f"{after_clip_norm:.6f}"
        )

        print(
            f"  Noise L2 norm               : "
            f"{noise_l2_norm:.6f}"
        )

    # ------------------------------------------------------------
    # Federated Averaging
    # ------------------------------------------------------------

    global_weights = federated_average(
        round_client_weights,
        round_sample_counts
    )

    # ------------------------------------------------------------
    # Evaluate updated global model
    # ------------------------------------------------------------

    global_loss, global_accuracy, global_precision, global_recall, global_f1, global_report, global_matrix = (
    evaluate_global_model(
        global_weights
    )
   )

    print("\nGlobal model:")
    print(
        f"  Test loss     : {global_loss:.4f}"
    )

    print(
        f"  Test accuracy : "
        f"{global_accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Store round history
    # ------------------------------------------------------------

    federated_history["rounds"].append(
        round_number
    )

    federated_history["global_accuracy"].append(
        global_accuracy
    )

    federated_history["global_loss"].append(
        global_loss
    )

    federated_history["client_metrics"].append(
        round_client_metrics
    )

    federated_history["privacy_metrics"].append(
        round_privacy_metrics
    )


print("\n" + "=" * 70)
print("FEDERATED TRAINING COMPLETED")
print("=" * 70)
# ================================================================
# SAVE FINAL FEDERATED MODEL AND RESULTS
# ================================================================

print("\n" + "=" * 70)
print("SAVING FINAL FEDERATED MODEL AND RESULTS")
print("=" * 70)

# ------------------------------------------------
# Create final model and assign global weights
# ------------------------------------------------

final_federated_model = build_fusion_model(
    learning_rate=LEARNING_RATE
)

final_federated_model.set_weights(
    global_weights
)


# ------------------------------------------------
# Save final federated model
# ------------------------------------------------

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "privacy_federated_fusion_final.keras"
)

final_federated_model.save(
    FINAL_MODEL_PATH
)

print("\nFinal model saved to:")
print(FINAL_MODEL_PATH)


# ------------------------------------------------
# Save federated history as JSON
# ------------------------------------------------

HISTORY_PATH = os.path.join(
    FUSION_RESULTS_DIR,
    "privacy_federated_history.json"
)

# Convert NumPy values to normal Python values
def convert_to_json_serializable(obj):

    if isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            convert_to_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    return obj


json_history = convert_to_json_serializable(
    federated_history
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    import json

    json.dump(
        json_history,
        file,
        indent=4
    )

print("\nFederated history saved to:")
print(HISTORY_PATH)


# ================================================================
# CREATE HUMAN-READABLE RESULTS FILE
# ================================================================

RESULTS_PATH = os.path.join(
    FUSION_RESULTS_DIR,
    "privacy_federated_results.txt"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PRIVACY-AWARE FEDERATED OPTIMIZED FUSION RESULTS\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Configuration\n"
    )

    file.write(
        f"Number of clients: {NUM_CLIENTS}\n"
    )

    file.write(
        f"Federated rounds: {FEDERATED_ROUNDS}\n"
    )

    file.write(
        f"Local epochs: {LOCAL_EPOCHS}\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    file.write(
        f"Clipping norm: {CLIP_NORM}\n"
    )

    file.write(
        f"Noise multiplier: {NOISE_MULTIPLIER}\n\n"
    )

    file.write(
        "Round-wise Global Results\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for i, round_number in enumerate(
        federated_history["rounds"]
    ):

        accuracy = (
            federated_history[
                "global_accuracy"
            ][i]
        )

        loss = (
            federated_history[
                "global_loss"
            ][i]
        )

        file.write(
            f"Round {round_number}: "
            f"Accuracy = {accuracy * 100:.2f}%, "
            f"Loss = {loss:.4f}\n"
        )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    final_accuracy = (
        federated_history[
            "global_accuracy"
        ][-1]
    )

    final_loss = (
        federated_history[
            "global_loss"
        ][-1]
    )

    file.write("\n")
    file.write(
        "Final Federated Result\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Final Test Accuracy: "
        f"{final_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Final Test Loss: "
        f"{final_loss:.4f}\n"
    )

    file.write("\n")
    file.write(
        "Privacy Mechanism\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        "Client updates are subjected to L2 norm clipping "
        "followed by Gaussian noise addition.\n"
    )

    file.write(
        "This implementation is treated as DP-style update "
        "perturbation; no formal epsilon-delta privacy accountant "
        "is claimed.\n"
    )


print("\nResults saved to:")
print(RESULTS_PATH)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL FEDERATED TRAINING SUMMARY")
print("=" * 70)

print(
    f"\nFinal federated test accuracy : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Final federated test loss     : "
    f"{final_loss:.4f}"
)

print("\nSaved files:")
print(f"1. {FINAL_MODEL_PATH}")
print(f"2. {HISTORY_PATH}")
print(f"3. {RESULTS_PATH}")

print("\n" + "=" * 70)
print("PRIVACY-AWARE FEDERATED EXPERIMENT COMPLETED")
print("=" * 70)