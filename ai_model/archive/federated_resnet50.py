import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "archive"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "Training"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "Testing"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


IMAGE_SIZE = (224, 224)

NUM_CLASSES = 4

NUM_CLIENTS = 3

FEDERATED_ROUNDS = 5

LOCAL_EPOCHS = 1

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print("=" * 70)
print("PRIVACY-AWARE FEDERATED RESNET50 TRAINING")
print("=" * 70)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nTesting directory:")
print(TEST_DIR)

print("\nNumber of clients:", NUM_CLIENTS)

print("Federated rounds:", FEDERATED_ROUNDS)

print("Local epochs:", LOCAL_EPOCHS)


# ============================================================
# VERIFY DATASET
# ============================================================

for class_name in [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]:

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    test_class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.isdir(train_class_dir):

        raise FileNotFoundError(
            f"Training class directory not found: "
            f"{train_class_dir}"
        )

    if not os.path.isdir(test_class_dir):

        raise FileNotFoundError(
            f"Testing class directory not found: "
            f"{test_class_dir}"
        )


# ============================================================
# COUNT DATA
# ============================================================

def count_images(directory):

    total = 0

    for root, dirs, files in os.walk(directory):

        for file_name in files:

            if file_name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):

                total += 1

    return total


train_total = count_images(
    TRAIN_DIR
)

test_total = count_images(
    TEST_DIR
)

print("\nDataset verification")

print("-----------------------------")

print(
    "Total training images:",
    train_total
)

print(
    "Total testing images:",
    test_total
)

if train_total != 5600:

    print(
        "\nWARNING: Expected 5600 training images "
        "based on your verified dataset."
    )

if test_total != 1600:

    print(
        "\nWARNING: Expected 1600 testing images "
        "based on your verified dataset."
    )


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="sparse",

    shuffle=False
)

print("\nTest class mapping:")

print(
    test_data.class_indices
)


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

expected_mapping = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

if test_data.class_indices != expected_mapping:

    raise ValueError(
        "\nClass mapping mismatch!\n"
        f"Expected: {expected_mapping}\n"
        f"Found: {test_data.class_indices}"
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

full_train_data = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="sparse",

    shuffle=True,

    seed=42
)

print("\nTraining class mapping:")

print(
    full_train_data.class_indices
)


if full_train_data.class_indices != expected_mapping:

    raise ValueError(
        "\nTraining class mapping mismatch!\n"
        f"Expected: {expected_mapping}\n"
        f"Found: {full_train_data.class_indices}"
    )


# ============================================================
# LOAD ALL TRAINING DATA INTO MEMORY
# ============================================================

print("\nLoading training images...")

x_train = []
y_train = []

for batch_index in range(
    len(full_train_data)
):

    images, labels = full_train_data[
        batch_index
    ]

    x_train.append(
        images
    )

    y_train.append(
        labels
    )


x_train = np.concatenate(
    x_train,
    axis=0
)

y_train = np.concatenate(
    y_train,
    axis=0
)

print(
    "Training array shape:",
    x_train.shape
)

print(
    "Training labels shape:",
    y_train.shape
)


# ============================================================
# CREATE CLIENT DATA
# ============================================================

print("\nCreating federated clients...")

indices = np.arange(
    len(x_train)
)

np.random.seed(42)

np.random.shuffle(
    indices
)

client_indices = np.array_split(
    indices,
    NUM_CLIENTS
)

clients = []

for client_id, indices_for_client in enumerate(
    client_indices,
    start=1
):

    client_x = x_train[
        indices_for_client
    ]

    client_y = y_train[
        indices_for_client
    ]

    clients.append(
        (
            client_x,
            client_y
        )
    )

    print(
        f"Client {client_id}: "
        f"{len(client_x)} images"
    )


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():

    print("\nCreating ResNet50 model...")

    backbone = ResNet50(

        weights="imagenet",

        include_top=False,

        input_shape=(
            224,
            224,
            3
        )

    )

    backbone.trainable = False

    inputs = layers.Input(
        shape=(
            224,
            224,
            3
        )
    )

    features = backbone(
        inputs,
        training=False
    )

    pooled = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d_1"
    )(
        features
    )

    dense_features = layers.Dense(
        256,
        activation="relu",
        name="feature_dense"
    )(
        pooled
    )

    dropout = layers.Dropout(
        0.5,
        name="feature_dropout"
    )(
        dense_features
    )

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classifier"
    )(
        dropout
    )

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )

    return model


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

global_model = create_model()

print("\nGlobal model created.")

global_model.summary()


# ============================================================
# FEDERATED WEIGHT AGGREGATION
# ============================================================

def federated_average(
    client_weights,
    client_sizes
):

    total_samples = sum(
        client_sizes
    )

    new_weights = []

    for layer_weights in zip(
        *client_weights
    ):

        weighted_sum = np.zeros_like(
            layer_weights[0]
        )

        for weights, size in zip(
            layer_weights,
            client_sizes
        ):

            weighted_sum += (
                weights
                * (
                    size
                    / total_samples
                )
            )

        new_weights.append(
            weighted_sum
        )

    return new_weights


# ============================================================
# FEDERATED TRAINING
# ============================================================

round_accuracies = []

round_losses = []


for round_number in range(
    1,
    FEDERATED_ROUNDS + 1
):

    print("\n")
    print("=" * 70)

    print(
        f"FEDERATED ROUND {round_number}"
    )

    print("=" * 70)


    client_weights = []

    client_sizes = []


    # ========================================================
    # CLIENT TRAINING
    # ========================================================

    for client_id, (
        client_x,
        client_y
    ) in enumerate(
        clients,
        start=1
    ):

        print(
            f"\nClient {client_id} "
            f"local training..."
        )

        client_model = create_model()

        client_model.set_weights(
            global_model.get_weights()
        )

        client_model.fit(

            client_x,

            client_y,

            batch_size=BATCH_SIZE,

            epochs=LOCAL_EPOCHS,

            verbose=1

        )

        client_weights.append(
            client_model.get_weights()
        )

        client_sizes.append(
            len(client_x)
        )

        tf.keras.backend.clear_session()


    # ========================================================
    # FEDAVG
    # ========================================================

    print(
        "\nAggregating client models using FedAvg..."
    )

    new_global_weights = federated_average(

        client_weights,

        client_sizes

    )

    global_model.set_weights(
        new_global_weights
    )


    # ========================================================
    # GLOBAL EVALUATION
    # ========================================================

    print(
        "\nEvaluating global model..."
    )

    loss, accuracy = global_model.evaluate(

        test_data,

        verbose=1

    )

    round_losses.append(
        float(loss)
    )

    round_accuracies.append(
        float(accuracy)
    )

    print(
        f"\nRound {round_number} "
        f"Global Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FEDERATED MODEL EVALUATION")
print("=" * 70)

final_loss, final_accuracy = global_model.evaluate(

    test_data,

    verbose=1

)

print(
    f"\nFinal Test Loss: "
    f"{final_loss:.6f}"
)

print(
    f"Final Test Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "federated_resnet50_final.keras"

)

global_model.save(
    final_model_path
)

print(
    "\nFederated model saved:"
)

print(
    final_model_path
)


# ============================================================
# SAVE TRAINING RESULTS
# ============================================================

results_path = os.path.join(

    MODEL_DIR,

    "federated_results.txt"

)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Federated ResNet50 Results\n"
    )

    file.write(
        "==========================\n\n"
    )

    file.write(
        f"Training images: {train_total}\n"
    )

    file.write(
        f"Testing images: {test_total}\n"
    )

    file.write(
        f"Clients: {NUM_CLIENTS}\n"
    )

    file.write(
        f"Federated rounds: "
        f"{FEDERATED_ROUNDS}\n"
    )

    file.write(
        f"Local epochs: "
        f"{LOCAL_EPOCHS}\n"
    )

    file.write(
        f"Final test loss: "
        f"{final_loss:.6f}\n"
    )

    file.write(
        f"Final test accuracy: "
        f"{final_accuracy * 100:.2f}%\n"
    )

    file.write(
        "\nRound results:\n"
    )

    for i in range(
        len(round_accuracies)
    ):

        file.write(
            f"Round {i + 1}: "
            f"Accuracy = "
            f"{round_accuracies[i] * 100:.2f}%, "
            f"Loss = "
            f"{round_losses[i]:.6f}\n"
        )


print(
    "\nResults saved:"
)

print(
    results_path
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 70)

print(
    "FEDERATED TRAINING COMPLETED SUCCESSFULLY"
)

print("=" * 70)