import os
import tensorflow as tf
import matplotlib.pyplot as plt

# ==============================
# CNN BASELINE MODEL
# ==============================

print("=" * 60)
print("CNN BASELINE MODEL")
print("=" * 60)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "Training")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# ==============================
# LOAD TRAINING DATA
# ==============================

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(i, ":", class_name)

print("\nNumber of classes:", len(class_names))

# ==============================
# NORMALIZATION
# ==============================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_data = validation_data.map(
    lambda x, y: (normalization_layer(x), y)
)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)

# ==============================
# CNN MODEL
# ==============================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Convolution Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Feature extraction
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Dropout
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# ==============================
# MODEL CHECKPOINT
# ==============================

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_path = os.path.join(
    MODEL_DIR,
    "cnn_baseline_best.keras"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ==============================
# TRAIN MODEL
# ==============================

print("\n" + "=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# ==============================
# SAVE FINAL MODEL
# ==============================

final_model_path = os.path.join(
    MODEL_DIR,
    "cnn_baseline_final.keras"
)

model.save(final_model_path)

print("\n" + "=" * 60)
print("CNN TRAINING COMPLETED")
print("=" * 60)

print("Best model saved at:")
print(checkpoint_path)

print("\nFinal model saved at:")
print(final_model_path)

# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("CNN Baseline Training and Validation Accuracy")

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    MODEL_DIR,
    "cnn_accuracy.png"
)

plt.savefig(accuracy_path)

print("\nAccuracy graph saved at:")
print(accuracy_path)

# ==============================
# LOSS GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("CNN Baseline Training and Validation Loss")

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    MODEL_DIR,
    "cnn_loss.png"
)

plt.savefig(loss_path)

print("\nLoss graph saved at:")
print(loss_path)

print("\nCNN baseline process completed successfully!")