import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# MEMBER 1 - MRI IMAGE PREPROCESSING
# ============================================================

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

# Image configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("=" * 60)
print("        MRI IMAGE PREPROCESSING")
print("=" * 60)

print("\nTraining path:")
print(TRAIN_DIR)

print("\nTesting path:")
print(TEST_DIR)

# Check folders
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Testing folder not found: {TEST_DIR}")

# ============================================================
# TRAINING + VALIDATION
# ============================================================
# Rescaling converts pixel values:
# 0-255  --->  0-1
#
# Augmentation is applied ONLY to training images.

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20,

    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,

    horizontal_flip=True,
    fill_mode="nearest"
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# ============================================================
# TESTING
# ============================================================
# IMPORTANT:
# No augmentation is applied to the testing dataset.

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nClass mapping:")
print(train_generator.class_indices)

print("\nTraining images:", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Testing images:", test_generator.samples)

print("\nImage size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nTraining distribution:")

for class_name, class_index in train_generator.class_indices.items():
    count = sum(
        train_generator.classes == class_index
    )
    print(f"{class_name:15}: {count}")

print("\nValidation distribution:")

for class_name, class_index in validation_generator.class_indices.items():
    count = sum(
        validation_generator.classes == class_index
    )
    print(f"{class_name:15}: {count}")

print("\nTesting distribution:")

for class_name, class_index in test_generator.class_indices.items():
    count = sum(
        test_generator.classes == class_index
    )
    print(f"{class_name:15}: {count}")

# ============================================================
# VERIFY ONE BATCH
# ============================================================

images, labels = next(train_generator)

print("\n" + "=" * 60)
print("PREPROCESSING VERIFICATION")
print("=" * 60)

print("\nBatch image shape:", images.shape)
print("Batch label shape:", labels.shape)

print("Minimum pixel value:", images.min())
print("Maximum pixel value:", images.max())

print("\nPreprocessing completed successfully!")