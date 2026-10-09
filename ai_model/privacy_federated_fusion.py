# ============================================================
# PRIVACY-AWARE FEDERATED FEATURE FUSION
# Brain Tumor MRI Classification
# ============================================================

import os
import json
import random
import numpy as np
import tensorflow as tf
import cv2
from sklearn.decomposition import PCA

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PROJECT PATHS
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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "fusion_results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# MODEL / FEDERATED CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

NUM_CLASSES = 4

NUM_CLIENTS = 3

FEDERATED_ROUNDS = 5

LOCAL_EPOCHS = 1

BATCH_SIZE = 32

LEARNING_RATE = 0.001


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

# Compact dimensions for the three handcrafted feature groups

INTENSITY_FEATURE_DIM = 32

TEXTURE_FEATURE_DIM = 32

SPATIAL_FEATURE_DIM = 32

# ResNet50 deep feature dimension after projection

DEEP_FEATURE_DIM = 128

# Final fused representation

FUSED_FEATURE_DIM = 128


# ============================================================
# PRIVACY CONFIGURATION
# ============================================================

# Client-update clipping threshold

CLIP_NORM = 1.0

# Gaussian noise multiplier.
# This will be measured and reported as part of the
# privacy-utility experiment.

NOISE_MULTIPLIER = 0.01


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]

CLASS_MAPPING = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("PRIVACY-AWARE FEDERATED FEATURE FUSION")
print("Brain Tumor MRI Classification")
print("=" * 70)

print("\nProject directory:")
print(BASE_DIR)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nTesting directory:")
print(TEST_DIR)

print("\nNumber of clients:", NUM_CLIENTS)

print("Federated rounds:", FEDERATED_ROUNDS)

print("Local epochs:", LOCAL_EPOCHS)

print("\nFeature configuration:")
print("Intensity features:", INTENSITY_FEATURE_DIM)
print("Texture features:", TEXTURE_FEATURE_DIM)
print("Spatial features:", SPATIAL_FEATURE_DIM)
print("Deep features:", DEEP_FEATURE_DIM)
print("Fused features:", FUSED_FEATURE_DIM)

print("\nPrivacy configuration:")
print("Clip norm:", CLIP_NORM)
print("Noise multiplier:", NOISE_MULTIPLIER)
# ============================================================
# DATASET INDEXING
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


def collect_image_paths(directory):
    """
    Collect image paths and their class labels.
    The class mapping is fixed so that the same class order
    is maintained throughout training and evaluation.
    """

    image_paths = []
    labels = []

    for class_name, class_index in CLASS_MAPPING.items():

        class_directory = os.path.join(
            directory,
            class_name
        )

        if not os.path.isdir(class_directory):

            raise FileNotFoundError(
                f"Class directory not found: {class_directory}"
            )

        for root, _, files in os.walk(
            class_directory
        ):

            for file_name in sorted(files):

                if file_name.lower().endswith(
                    VALID_EXTENSIONS
                ):

                    image_paths.append(
                        os.path.join(
                            root,
                            file_name
                        )
                    )

                    labels.append(
                        class_index
                    )

    return (
        np.asarray(image_paths),
        np.asarray(labels, dtype=np.int32)
    )


# ============================================================
# COLLECT TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("INDEXING TRAINING DATA")
print("=" * 70)

train_paths, train_labels = collect_image_paths(
    TRAIN_DIR
)

print(
    "\nTotal training images:",
    len(train_paths)
)


# ============================================================
# COLLECT TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("INDEXING TEST DATA")
print("=" * 70)

test_paths, test_labels = collect_image_paths(
    TEST_DIR
)

print(
    "\nTotal testing images:",
    len(test_paths)
)


# ============================================================
# VERIFY DATASET SIZE
# ============================================================

if len(train_paths) != 5600:

    raise ValueError(
        f"Expected 5600 training images, "
        f"but found {len(train_paths)}."
    )


if len(test_paths) != 1600:

    raise ValueError(
        f"Expected 1600 testing images, "
        f"but found {len(test_paths)}."
    )


# ============================================================
# VERIFY CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

for class_name, class_index in CLASS_MAPPING.items():

    count = int(
        np.sum(
            train_labels == class_index
        )
    )

    print(
        f"{class_name:12s}: {count}"
    )


print("\nTesting class distribution:")

for class_name, class_index in CLASS_MAPPING.items():

    count = int(
        np.sum(
            test_labels == class_index
        )
    )

    print(
        f"{class_name:12s}: {count}"
    )


# ============================================================
# VERIFY CLASS COUNTS
# ============================================================

expected_train_count = 1400
expected_test_count = 400

for class_name, class_index in CLASS_MAPPING.items():

    train_count = int(
        np.sum(
            train_labels == class_index
        )
    )

    test_count = int(
        np.sum(
            test_labels == class_index
        )
    )

    if train_count != expected_train_count:

        raise ValueError(
            f"Unexpected training count for "
            f"{class_name}: {train_count}"
        )

    if test_count != expected_test_count:

        raise ValueError(
            f"Unexpected testing count for "
            f"{class_name}: {test_count}"
        )


print("\nDataset indexing completed successfully.")
# ============================================================
# HANDCRAFTED FEATURE EXTRACTION
# Intensity + Texture + Spatial
# ============================================================


def load_grayscale_image(image_path):
    """
    Load an MRI image as grayscale and resize it to 224 x 224.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ============================================================
# INTENSITY FEATURES
# ============================================================

def extract_intensity_features(gray):
    """
    Extract intensity-distribution features.

    16 histogram bins
    + 16 statistical/percentile features
    = 32 features
    """

    # --------------------------------------------------------
    # Normalized intensity histogram
    # --------------------------------------------------------

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [16],
        [0.0, 1.0]
    ).flatten()

    histogram = histogram / (
        np.sum(histogram) + 1e-8
    )

    # --------------------------------------------------------
    # Statistical features
    # --------------------------------------------------------

    pixels = gray.reshape(-1)

    statistics = np.array([
        np.mean(pixels),
        np.std(pixels),
        np.min(pixels),
        np.max(pixels),
        np.median(pixels),

        np.percentile(pixels, 5),
        np.percentile(pixels, 10),
        np.percentile(pixels, 20),
        np.percentile(pixels, 30),
        np.percentile(pixels, 40),
        np.percentile(pixels, 50),
        np.percentile(pixels, 60),
        np.percentile(pixels, 70),
        np.percentile(pixels, 80),
        np.percentile(pixels, 90),
        np.percentile(pixels, 95)
    ], dtype=np.float32)

    features = np.concatenate([
        histogram,
        statistics
    ])

    return features.astype(
        np.float32
    )


# ============================================================
# TEXTURE FEATURES
# ============================================================

def extract_lbp_histogram(gray):
    """
    Extract a compact Local Binary Pattern representation.

    The 8-neighbourhood LBP code is quantized into
    16 bins to keep the representation compact.
    """

    center = gray[1:-1, 1:-1]

    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2]
    ]

    lbp = np.zeros_like(
        center,
        dtype=np.uint8
    )

    for bit, neighbor in enumerate(
        neighbors
    ):

        lbp |= (
            (neighbor >= center).astype(
                np.uint8
            ) << bit
        )

    # Quantize 256 possible LBP codes
    # into 16 compact bins.

    quantized_lbp = (
        lbp.astype(np.int32) // 16
    )

    histogram = np.bincount(
        quantized_lbp.reshape(-1),
        minlength=16
    ).astype(
        np.float32
    )

    histogram = histogram / (
        np.sum(histogram) + 1e-8
    )

    return histogram


def extract_gabor_features(gray):
    """
    Extract Gabor texture responses using
    four different orientations.

    Each orientation contributes:
    mean + standard deviation
    = 8 features.
    """

    features = []

    orientations = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    for theta in orientations:

        kernel = cv2.getGaborKernel(
            (21, 21),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )

        response = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel
        )

        features.append(
            float(np.mean(response))
        )

        features.append(
            float(np.std(response))
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


def extract_texture_features(gray):
    """
    Combine LBP, Gabor and gradient statistics.

    16 LBP features
    + 8 Gabor features
    + 8 gradient features
    = 32 texture features.
    """

    # --------------------------------------------------------
    # LBP
    # --------------------------------------------------------

    lbp_features = extract_lbp_histogram(
        gray
    )

    # --------------------------------------------------------
    # Gabor
    # --------------------------------------------------------

    gabor_features = extract_gabor_features(
        gray
    )

    # --------------------------------------------------------
    # Gradient statistics
    # --------------------------------------------------------

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = cv2.magnitude(
        gradient_x,
        gradient_y
    )

    gradient_statistics = np.array([
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.min(gradient_magnitude),
        np.max(gradient_magnitude),

        np.percentile(
            gradient_magnitude,
            25
        ),

        np.percentile(
            gradient_magnitude,
            50
        ),

        np.percentile(
            gradient_magnitude,
            75
        ),

        np.percentile(
            gradient_magnitude,
            90
        )
    ], dtype=np.float32)

    features = np.concatenate([
        lbp_features,
        gabor_features,
        gradient_statistics
    ])

    return features.astype(
        np.float32
    )


# ============================================================
# SPATIAL FEATURES
# ============================================================

def extract_spatial_features(gray):
    """
    Extract spatial information using a 4 x 4 grid.

    For each grid cell:
    - mean intensity
    - mean gradient magnitude

    16 + 16 = 32 spatial features.
    """

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = cv2.magnitude(
        gradient_x,
        gradient_y
    )

    intensity_grid = []

    gradient_grid = []

    height, width = gray.shape

    cell_height = height // 4
    cell_width = width // 4

    for row in range(4):

        for col in range(4):

            y_start = row * cell_height
            y_end = (
                (row + 1) * cell_height
            )

            x_start = col * cell_width
            x_end = (
                (col + 1) * cell_width
            )

            intensity_cell = gray[
                y_start:y_end,
                x_start:x_end
            ]

            gradient_cell = gradient_magnitude[
                y_start:y_end,
                x_start:x_end
            ]

            intensity_grid.append(
                np.mean(
                    intensity_cell
                )
            )

            gradient_grid.append(
                np.mean(
                    gradient_cell
                )
            )

    features = np.concatenate([
        np.asarray(
            intensity_grid,
            dtype=np.float32
        ),

        np.asarray(
            gradient_grid,
            dtype=np.float32
        )
    ])

    return features.astype(
        np.float32
    )


# ============================================================
# COMPLETE HANDCRAFTED FEATURE EXTRACTION
# ============================================================

def extract_handcrafted_features(
    image_path
):
    """
    Extract all three handcrafted feature groups
    from one MRI image.
    """

    gray = load_grayscale_image(
        image_path
    )

    intensity_features = (
        extract_intensity_features(
            gray
        )
    )

    texture_features = (
        extract_texture_features(
            gray
        )
    )

    spatial_features = (
        extract_spatial_features(
            gray
        )
    )

    return (
        intensity_features,
        texture_features,
        spatial_features
    )


# ============================================================
# FEATURE DIMENSION TEST
# ============================================================

print("\n" + "=" * 70)
print("TESTING HANDCRAFTED FEATURE EXTRACTION")
print("=" * 70)

sample_image = train_paths[0]

(
    sample_intensity,
    sample_texture,
    sample_spatial
) = extract_handcrafted_features(
    sample_image
)

print(
    "\nSample image:",
    sample_image
)

print(
    "Intensity feature shape:",
    sample_intensity.shape
)

print(
    "Texture feature shape:",
    sample_texture.shape
)

print(
    "Spatial feature shape:",
    sample_spatial.shape
)


# ============================================================
# VERIFY FEATURE DIMENSIONS
# ============================================================

if sample_intensity.shape[0] != INTENSITY_FEATURE_DIM:

    raise ValueError(
        "Intensity feature dimension mismatch."
    )


if sample_texture.shape[0] != TEXTURE_FEATURE_DIM:

    raise ValueError(
        "Texture feature dimension mismatch."
    )


if sample_spatial.shape[0] != SPATIAL_FEATURE_DIM:

    raise ValueError(
        "Spatial feature dimension mismatch."
    )


print(
    "\nHandcrafted feature extraction test "
    "completed successfully."
)
# ============================================================
# RESNET50 DEEP FEATURE EXTRACTOR
# ============================================================

print("\n" + "=" * 70)
print("BUILDING RESNET50 DEEP FEATURE EXTRACTOR")
print("=" * 70)


def create_resnet_feature_extractor():
    """
    Create a frozen ImageNet ResNet50 backbone.

    The backbone is used only for feature extraction.
    Federated training will operate on the compact
    fused representation rather than repeatedly
    retraining the complete CNN.
    """

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
        ),
        name="image_input"
    )

    features = backbone(
        inputs,
        training=False
    )

    pooled_features = (
        layers.GlobalAveragePooling2D(
            name="resnet_global_average_pooling"
        )(
            features
        )
    )

        # Keep the original 2048-dimensional ResNet50
    # representation for PCA-based dimensionality reduction.

    projected_features = layers.Lambda(
        lambda x: x,
        name="deep_feature_projection"
    )(
        pooled_features
    )

    feature_extractor = models.Model(
        inputs=inputs,
        outputs=projected_features,
        name="resnet50_feature_extractor"
    )

    return feature_extractor

deep_feature_extractor = (
    create_resnet_feature_extractor()
)


print(
    "\nResNet50 deep feature extractor created."
)

print(
    "Output dimension:",
    deep_feature_extractor.output_shape
)
# ============================================================
# TEST RESNET50 DEEP FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("TESTING RESNET50 DEEP FEATURE EXTRACTION")
print("=" * 70)


def load_rgb_image(image_path):
    """
    Load and preprocess an MRI image for ResNet50.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(
        np.float32
    )

    image = preprocess_input(
        image
    )

    return image


sample_rgb = load_rgb_image(
    sample_image
)

sample_rgb_batch = np.expand_dims(
    sample_rgb,
    axis=0
)


sample_deep = (
    deep_feature_extractor.predict(
        sample_rgb_batch,
        verbose=0
    )
)


print(
    "\nSample deep feature shape:",
    sample_deep.shape
)


# ============================================================
# VERIFY DEEP FEATURE DIMENSION
# ============================================================

if sample_deep.shape[1] != 2048:

    raise ValueError(
        "ResNet50 raw deep feature dimension mismatch."
    )


print(
    "ResNet50 raw deep feature extraction "
    "test completed successfully."
)
# ============================================================
# FEATURE CACHE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CACHE CONFIGURATION")
print("=" * 70)


TRAIN_FEATURE_CACHE = os.path.join(
    RESULTS_DIR,
    "train_features.npz"
)

TEST_FEATURE_CACHE = os.path.join(
    RESULTS_DIR,
    "test_features.npz"
)


# ============================================================
# FEATURE EXTRACTION FOR DATASET
# ============================================================

def extract_dataset_features(
    image_paths,
    labels,
    dataset_name
):
    """
    Extract handcrafted and ResNet50 deep features
    for a complete dataset.

    Handcrafted features are extracted image-by-image,
    while ResNet50 deep features are extracted in
    batches for better CPU efficiency.
    """

    print("\n" + "-" * 70)
    print(
        f"Extracting features for {dataset_name}..."
    )
    print("-" * 70)

    intensity_features = []
    texture_features = []
    spatial_features = []
    deep_features = []

    total_images = len(
        image_paths
    )

    # --------------------------------------------------------
    # Temporary batch storage for ResNet50
    # --------------------------------------------------------

    rgb_batch = []

    for index, image_path in enumerate(
        image_paths
    ):

        # ----------------------------------------------------
        # Handcrafted features
        # ----------------------------------------------------

        (
            intensity,
            texture,
            spatial
        ) = extract_handcrafted_features(
            image_path
        )

        intensity_features.append(
            intensity
        )

        texture_features.append(
            texture
        )

        spatial_features.append(
            spatial
        )

        # ----------------------------------------------------
        # Prepare RGB image for ResNet50
        # ----------------------------------------------------

        rgb_image = load_rgb_image(
            image_path
        )

        rgb_batch.append(
            rgb_image
        )

        # ----------------------------------------------------
        # Process ResNet50 in batches
        # ----------------------------------------------------

        is_batch_full = (
            len(rgb_batch) >= BATCH_SIZE
        )

        is_last_image = (
            index == total_images - 1
        )

        if is_batch_full or is_last_image:

            batch_array = np.asarray(
                rgb_batch,
                dtype=np.float32
            )

            batch_deep_features = (
                deep_feature_extractor.predict(
                    batch_array,
                    verbose=0
                )
            )

            for feature in batch_deep_features:

                deep_features.append(
                    feature
                )

            rgb_batch = []

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        completed = index + 1

        if (
            completed % 100 == 0
            or completed == total_images
        ):

            print(
                f"{dataset_name}: "
                f"{completed}/{total_images} images processed"
            )

    # --------------------------------------------------------
    # Convert everything to NumPy arrays
    # --------------------------------------------------------

    return (
        np.asarray(
            intensity_features,
            dtype=np.float32
        ),

        np.asarray(
            texture_features,
            dtype=np.float32
        ),

        np.asarray(
            spatial_features,
            dtype=np.float32
        ),

        np.asarray(
            deep_features,
            dtype=np.float32
        ),

        np.asarray(
            labels,
            dtype=np.int32
        )
    )


# ============================================================
# CACHE TRAINING FEATURES
# ============================================================

if os.path.exists(
    TRAIN_FEATURE_CACHE
):

    print(
        "\nTraining feature cache already exists."
    )

    print(
        "Loading:",
        TRAIN_FEATURE_CACHE
    )

    train_cache = np.load(
        TRAIN_FEATURE_CACHE
    )

    train_intensity = train_cache[
        "intensity"
    ]

    train_texture = train_cache[
        "texture"
    ]

    train_spatial = train_cache[
        "spatial"
    ]

    train_deep = train_cache[
        "deep"
    ]

    train_cached_labels = train_cache[
        "labels"
    ]

else:

    print(
        "\nTraining feature cache not found."
    )

    print(
        "Creating training feature cache..."
    )

    (
        train_intensity,
        train_texture,
        train_spatial,
        train_deep,
        train_cached_labels
    ) = extract_dataset_features(
        train_paths,
        train_labels,
        "Training"
    )

    np.savez_compressed(
        TRAIN_FEATURE_CACHE,
        intensity=train_intensity,
        texture=train_texture,
        spatial=train_spatial,
        deep=train_deep,
        labels=train_cached_labels
    )

    print(
        "\nTraining features saved:"
    )

    print(
        TRAIN_FEATURE_CACHE
    )


# ============================================================
# CACHE TEST FEATURES
# ============================================================

if os.path.exists(
    TEST_FEATURE_CACHE
):

    print(
        "\nTesting feature cache already exists."
    )

    print(
        "Loading:",
        TEST_FEATURE_CACHE
    )

    test_cache = np.load(
        TEST_FEATURE_CACHE
    )

    test_intensity = test_cache[
        "intensity"
    ]

    test_texture = test_cache[
        "texture"
    ]

    test_spatial = test_cache[
        "spatial"
    ]

    test_deep = test_cache[
        "deep"
    ]

    test_cached_labels = test_cache[
        "labels"
    ]

else:

    print(
        "\nTesting feature cache not found."
    )

    print(
        "Creating testing feature cache..."
    )

    (
        test_intensity,
        test_texture,
        test_spatial,
        test_deep,
        test_cached_labels
    ) = extract_dataset_features(
        test_paths,
        test_labels,
        "Testing"
    )

    np.savez_compressed(
        TEST_FEATURE_CACHE,
        intensity=test_intensity,
        texture=test_texture,
        spatial=test_spatial,
        deep=test_deep,
        labels=test_cached_labels
    )

    print(
        "\nTesting features saved:"
    )

    print(
        TEST_FEATURE_CACHE
    )


# ============================================================
# CACHE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING FEATURE CACHE")
print("=" * 70)


print(
    "\nTraining intensity:",
    train_intensity.shape
)

print(
    "Training texture:",
    train_texture.shape
)

print(
    "Training spatial:",
    train_spatial.shape
)

print(
    "Training deep:",
    train_deep.shape
)

print(
    "\nTesting intensity:",
    test_intensity.shape
)

print(
    "Testing texture:",
    test_texture.shape
)

print(
    "Testing spatial:",
    test_spatial.shape
)

print(
    "Testing deep:",
    test_deep.shape
)


# ============================================================
# VERIFY TRAINING CACHE
# ============================================================

if train_intensity.shape != (
    len(train_paths),
    INTENSITY_FEATURE_DIM
):

    raise ValueError(
        "Training intensity cache shape mismatch."
    )


if train_texture.shape != (
    len(train_paths),
    TEXTURE_FEATURE_DIM
):

    raise ValueError(
        "Training texture cache shape mismatch."
    )


if train_spatial.shape != (
    len(train_paths),
    SPATIAL_FEATURE_DIM
):

    raise ValueError(
        "Training spatial cache shape mismatch."
    )

if train_deep.shape != (
    len(train_paths),
    2048
):

    raise ValueError(
        "Training raw deep cache shape mismatch."
    )



# ============================================================
# VERIFY TESTING CACHE
# ============================================================

if test_intensity.shape != (
    len(test_paths),
    INTENSITY_FEATURE_DIM
):

    raise ValueError(
        "Testing intensity cache shape mismatch."
    )


if test_texture.shape != (
    len(test_paths),
    TEXTURE_FEATURE_DIM
):

    raise ValueError(
        "Testing texture cache shape mismatch."
    )


if test_spatial.shape != (
    len(test_paths),
    SPATIAL_FEATURE_DIM
):

    raise ValueError(
        "Testing spatial cache shape mismatch."
    )


if test_deep.shape != (
    len(test_paths),
    2048
):

    raise ValueError(
        "Testing raw deep cache shape mismatch."
    )


# ============================================================
# VERIFY LABELS
# ============================================================

if not np.array_equal(
    train_cached_labels,
    train_labels
):

    raise ValueError(
        "Training cached labels do not match "
        "original labels."
    )


if not np.array_equal(
    test_cached_labels,
    test_labels
):

    raise ValueError(
        "Testing cached labels do not match "
        "original labels."
    )


print(
    "\nFeature cache validation completed successfully."
)

print(
    "\nTotal features per MRI:",
    (
        INTENSITY_FEATURE_DIM
        + TEXTURE_FEATURE_DIM
        + SPATIAL_FEATURE_DIM
        + DEEP_FEATURE_DIM
    )
)
# ============================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================

def fit_pca_and_transform(
    train_deep_features,
    test_deep_features
):
    """
    Reduce the 2048-dimensional ResNet50 features
    to a compact 128-dimensional representation.

    PCA is fitted ONLY on training data to avoid
    information leakage from the test set.
    """

    print("\n" + "=" * 70)
    print("PCA DIMENSIONALITY REDUCTION")
    print("=" * 70)

    print(
        "\nOriginal deep feature dimension:",
        train_deep_features.shape[1]
    )

    print(
        "Target deep feature dimension:",
        DEEP_FEATURE_DIM
    )

    pca = PCA(
        n_components=DEEP_FEATURE_DIM,
        svd_solver="randomized",
        random_state=SEED
    )

    # --------------------------------------------------------
    # Fit PCA only on training features
    # --------------------------------------------------------

    train_reduced = pca.fit_transform(
        train_deep_features
    )

    # --------------------------------------------------------
    # Transform test features using the same PCA
    # --------------------------------------------------------

    test_reduced = pca.transform(
        test_deep_features
    )

    explained_variance = float(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    print(
        "\nReduced training feature shape:",
        train_reduced.shape
    )

    print(
        "Reduced testing feature shape:",
        test_reduced.shape
    )

    print(
        f"Total explained variance: "
        f"{explained_variance * 100:.2f}%"
    )

    return (
        train_reduced.astype(np.float32),
        test_reduced.astype(np.float32),
        pca,
        explained_variance
    )
    # ============================================================
# APPLY PCA TO CACHED RESNET50 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("REDUCING RESNET50 FEATURES WITH PCA")
print("=" * 70)

(
    train_deep_reduced,
    test_deep_reduced,
    deep_pca,
    explained_variance
) = fit_pca_and_transform(
    train_deep,
    test_deep
)


# ============================================================
# VERIFY PCA OUTPUT
# ============================================================

print(
    "\nPCA training features:",
    train_deep_reduced.shape
)

print(
    "PCA testing features:",
    test_deep_reduced.shape
)


if train_deep_reduced.shape != (
    len(train_paths),
    DEEP_FEATURE_DIM
):

    raise ValueError(
        "PCA training feature shape mismatch."
    )


if test_deep_reduced.shape != (
    len(test_paths),
    DEEP_FEATURE_DIM
):

    raise ValueError(
        "PCA testing feature shape mismatch."
    )


# ============================================================
# SAVE PCA-REDUCED FEATURES
# ============================================================

PCA_TRAIN_CACHE = os.path.join(
    RESULTS_DIR,
    "train_features_pca.npz"
)

PCA_TEST_CACHE = os.path.join(
    RESULTS_DIR,
    "test_features_pca.npz"
)


np.savez_compressed(
    PCA_TRAIN_CACHE,
    intensity=train_intensity,
    texture=train_texture,
    spatial=train_spatial,
    deep=train_deep_reduced,
    labels=train_cached_labels
)


np.savez_compressed(
    PCA_TEST_CACHE,
    intensity=test_intensity,
    texture=test_texture,
    spatial=test_spatial,
    deep=test_deep_reduced,
    labels=test_cached_labels
)


print(
    "\nPCA-reduced training features saved:"
)

print(
    PCA_TRAIN_CACHE
)


print(
    "\nPCA-reduced testing features saved:"
)

print(
    PCA_TEST_CACHE
)


print(
    "\nExplained variance retained:",
    f"{explained_variance * 100:.2f}%"
)


print(
    "\nPCA dimensionality reduction completed successfully."
)
# ============================================================
# OPTIMIZED FEATURE FUSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING OPTIMIZED FEATURE FUSION MODEL")
print("=" * 70)


# ------------------------------------------------------------
# LOAD PCA-REDUCED FEATURES
# ------------------------------------------------------------

train_pca_data = np.load(PCA_TRAIN_CACHE)
test_pca_data = np.load(PCA_TEST_CACHE)

train_intensity_fusion = train_pca_data["intensity"].astype(np.float32)
train_texture_fusion = train_pca_data["texture"].astype(np.float32)
train_spatial_fusion = train_pca_data["spatial"].astype(np.float32)
train_deep_fusion = train_pca_data["deep"].astype(np.float32)
train_labels_fusion = train_pca_data["labels"].astype(np.int32)

test_intensity_fusion = test_pca_data["intensity"].astype(np.float32)
test_texture_fusion = test_pca_data["texture"].astype(np.float32)
test_spatial_fusion = test_pca_data["spatial"].astype(np.float32)
test_deep_fusion = test_pca_data["deep"].astype(np.float32)
test_labels_fusion = test_pca_data["labels"].astype(np.int32)


print("\nFusion feature shapes:")
print("Intensity:", train_intensity_fusion.shape)
print("Texture :", train_texture_fusion.shape)
print("Spatial :", train_spatial_fusion.shape)
print("Deep    :", train_deep_fusion.shape)


# ------------------------------------------------------------
# FEATURE STANDARDIZATION
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler


intensity_scaler = StandardScaler()
texture_scaler = StandardScaler()
spatial_scaler = StandardScaler()
deep_scaler = StandardScaler()


train_intensity_scaled = intensity_scaler.fit_transform(
    train_intensity_fusion
).astype(np.float32)

test_intensity_scaled = intensity_scaler.transform(
    test_intensity_fusion
).astype(np.float32)


train_texture_scaled = texture_scaler.fit_transform(
    train_texture_fusion
).astype(np.float32)

test_texture_scaled = texture_scaler.transform(
    test_texture_fusion
).astype(np.float32)


train_spatial_scaled = spatial_scaler.fit_transform(
    train_spatial_fusion
).astype(np.float32)

test_spatial_scaled = spatial_scaler.transform(
    test_spatial_fusion
).astype(np.float32)


train_deep_scaled = deep_scaler.fit_transform(
    train_deep_fusion
).astype(np.float32)

test_deep_scaled = deep_scaler.transform(
    test_deep_fusion
).astype(np.float32)


print("\nFeature standardization completed.")


# ------------------------------------------------------------
# OPTIMIZED GATED FEATURE FUSION
# ------------------------------------------------------------

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
        shape=(128,),
        name="deep_input"
    )


    # --------------------------------------------------------
    # BRANCH-SPECIFIC FEATURE PROJECTIONS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COMBINED REPRESENTATION FOR GATING
    # --------------------------------------------------------

    combined_features = layers.Concatenate(
        name="combined_branch_features"
    )([
        intensity_branch,
        texture_branch,
        spatial_branch,
        deep_branch
    ])


    # --------------------------------------------------------
    # LEARNABLE GATING NETWORK
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SPLIT GATE WEIGHTS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # APPLY LEARNABLE WEIGHTS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FINAL FUSED REPRESENTATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

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


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    return model


fusion_model = build_optimized_fusion_model()


print("\nOptimized fusion model created successfully.")
fusion_model.summary()
# ============================================================
# CENTRALIZED OPTIMIZED FUSION TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CENTRALIZED OPTIMIZED FEATURE-FUSION MODEL")
print("=" * 70)


# ------------------------------------------------------------
# MODEL SAVE PATHS
# ------------------------------------------------------------

FUSION_BEST_MODEL = os.path.join(
    MODEL_DIR,
    "optimized_fusion_best.keras"
)

FUSION_FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "optimized_fusion_final.keras"
)

FUSION_HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "optimized_fusion_history.json"
)


# ------------------------------------------------------------
# TRAINING CALLBACKS
# ------------------------------------------------------------

fusion_callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        FUSION_BEST_MODEL,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ------------------------------------------------------------
# TRAIN CENTRALIZED MODEL
# ------------------------------------------------------------

fusion_history = fusion_model.fit(

    [
        train_intensity_scaled,
        train_texture_scaled,
        train_spatial_scaled,
        train_deep_scaled
    ],

    train_labels_fusion,

    validation_split=0.20,

    epochs=30,

    batch_size=BATCH_SIZE,

    callbacks=fusion_callbacks,

    verbose=1
)


# ------------------------------------------------------------
# SAVE FINAL MODEL
# ------------------------------------------------------------

fusion_model.save(FUSION_FINAL_MODEL)


# ------------------------------------------------------------
# SAVE TRAINING HISTORY
# ------------------------------------------------------------

with open(FUSION_HISTORY_PATH, "w") as f:

    json.dump(
        fusion_history.history,
        f,
        indent=4
    )


print("\nCentralized optimized fusion training completed.")

print(
    "\nBest model saved:"
)

print(
    FUSION_BEST_MODEL
)

print(
    "\nFinal model saved:"
)

print(
    FUSION_FINAL_MODEL
)

print(
    "\nTraining history saved:"
)

print(
    FUSION_HISTORY_PATH
)
# ============================================================
# EVALUATE CENTRALIZED OPTIMIZED FUSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING CENTRALIZED OPTIMIZED FEATURE-FUSION MODEL")
print("=" * 70)


# ------------------------------------------------------------
# LOAD BEST MODEL
# ------------------------------------------------------------

best_fusion_model = tf.keras.models.load_model(
    FUSION_BEST_MODEL,
    safe_mode=False
)

print("\nBest optimized fusion model loaded successfully.")


# ------------------------------------------------------------
# TEST SET EVALUATION
# ------------------------------------------------------------

test_loss, test_accuracy = best_fusion_model.evaluate(
    [
        test_intensity_scaled,
        test_texture_scaled,
        test_spatial_scaled,
        test_deep_scaled
    ],
    test_labels_fusion,
    batch_size=BATCH_SIZE,
    verbose=1
)


print("\n" + "-" * 70)
print("CENTRALIZED OPTIMIZED FUSION TEST RESULTS")
print("-" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# GENERATE PREDICTIONS
# ------------------------------------------------------------

test_probabilities = best_fusion_model.predict(
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


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


classification_report_text = classification_report(
    test_labels_fusion,
    test_predictions,
    target_names=CLASS_NAMES,
    digits=4
)


print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(classification_report_text)


# ------------------------------------------------------------
# OVERALL METRICS
# ------------------------------------------------------------

precision = precision_score(
    test_labels_fusion,
    test_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    test_labels_fusion,
    test_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    test_labels_fusion,
    test_predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "-" * 70)
print("OVERALL TEST METRICS")
print("-" * 70)

print(
    f"Accuracy  : {test_accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    test_labels_fusion,
    test_predictions
)


print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print(cm)


# ------------------------------------------------------------
# SAVE CONFUSION MATRIX
# ------------------------------------------------------------

import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title(
    "Centralized Optimized Feature Fusion - Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=30
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

FUSION_CM_PATH = os.path.join(
    RESULTS_DIR,
    "optimized_fusion_confusion_matrix.png"
)

plt.savefig(
    FUSION_CM_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# SAVE FINAL EVALUATION RESULTS
# ------------------------------------------------------------

FUSION_EVALUATION_PATH = os.path.join(
    RESULTS_DIR,
    "optimized_fusion_evaluation.txt"
)


with open(FUSION_EVALUATION_PATH, "w") as f:

    f.write(
        "CENTRALIZED OPTIMIZED FEATURE-FUSION MODEL\n"
    )

    f.write(
        "============================================\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.6f}\n"
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

    f.write(
        classification_report_text
    )

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        "----------------\n"
    )

    f.write(
        str(cm)
    )


print("\n" + "=" * 70)
print("CENTRALIZED FUSION EVALUATION COMPLETED")
print("=" * 70)

print(
    "\nConfusion matrix saved:"
)

print(
    FUSION_CM_PATH
)

print(
    "\nEvaluation results saved:"
)

print(
    FUSION_EVALUATION_PATH
)
# ======================================================================
# NON-IID CLIENT PARTITIONING
# ======================================================================

print("\n" + "=" * 70)
print("CREATING NON-IID FEDERATED CLIENT PARTITIONS")
print("=" * 70)

from collections import Counter


def create_non_iid_clients(
    intensity_features,
    texture_features,
    spatial_features,
    deep_features,
    labels
):
    """
    Create heterogeneous (non-IID) client datasets.

    Each client receives a different class distribution to simulate
    realistic differences between healthcare institutions.
    """

    # Class-specific sample indices
    class_indices = {
        class_id: np.where(labels == class_id)[0]
        for class_id in range(NUM_CLASSES)
    }

    # Shuffle samples within each class
    rng = np.random.default_rng(SEED)

    for class_id in class_indices:
        rng.shuffle(class_indices[class_id])

    # --------------------------------------------------------------
    # Define heterogeneous class distributions.
    #
    # Client 1: more Glioma + Meningioma
    # Client 2: more Meningioma + Pituitary
    # Client 3: more No Tumor + Pituitary
    #
    # All samples are used exactly once.
    # --------------------------------------------------------------

    client_class_counts = {
        0: {0: 800, 1: 400, 2: 100, 3: 100},
        1: {0: 300, 1: 700, 2: 300, 3: 700},
        2: {0: 300, 1: 300, 2: 1000, 3: 600},
    }

    client_indices = {}

    for client_id in range(NUM_CLIENTS):
        selected_indices = []

        for class_id in range(NUM_CLASSES):
            count = client_class_counts[client_id][class_id]

            selected = class_indices[class_id][:count]

            # Remove selected samples so another client
            # cannot receive the same samples.
            class_indices[class_id] = class_indices[class_id][count:]

            selected_indices.extend(selected.tolist())

        rng.shuffle(selected_indices)
        client_indices[client_id] = np.array(
            selected_indices,
            dtype=np.int32
        )

    # --------------------------------------------------------------
    # Verify that every training sample was assigned exactly once.
    # --------------------------------------------------------------

    all_assigned = np.concatenate(
        [client_indices[i] for i in range(NUM_CLIENTS)]
    )

    assert len(all_assigned) == len(labels), (
        "Incorrect number of assigned samples."
    )

    assert len(np.unique(all_assigned)) == len(labels), (
        "Duplicate samples detected between clients."
    )

    assert set(all_assigned.tolist()) == set(
        range(len(labels))
    ), (
        "Some training samples were not assigned to any client."
    )

    # --------------------------------------------------------------
    # Build client feature dictionaries
    # --------------------------------------------------------------

    clients = {}

    for client_id in range(NUM_CLIENTS):

        idx = client_indices[client_id]

        clients[client_id] = {
            "intensity": intensity_features[idx],
            "texture": texture_features[idx],
            "spatial": spatial_features[idx],
            "deep": deep_features[idx],
            "labels": labels[idx],
        }

    # --------------------------------------------------------------
    # Display client distributions
    # --------------------------------------------------------------

    print("\nNon-IID client distributions:")

    for client_id in range(NUM_CLIENTS):

        client_labels = clients[client_id]["labels"]
        counts = Counter(client_labels.tolist())

        print(f"\nClient {client_id + 1}")
        print("-" * 40)

        for class_id, class_name in enumerate(CLASS_NAMES):
            print(
                f"{class_name:<15}: "
                f"{counts.get(class_id, 0):>4}"
            )

        print(
            f"{'Total':<15}: "
            f"{len(client_labels):>4}"
        )

    print("\nNon-IID partitioning completed successfully.")

    return clients


# ----------------------------------------------------------------------
# Load PCA-reduced training features
# ----------------------------------------------------------------------

train_pca_data = np.load(
    os.path.join(
        FUSION_RESULTS_DIR,
        "train_features_pca.npz"
    )
)

train_intensity = train_pca_data["intensity"]
train_texture = train_pca_data["texture"]
train_spatial = train_pca_data["spatial"]
train_deep = train_pca_data["deep"]
train_labels = train_pca_data["labels"]


# ----------------------------------------------------------------------
# Create non-IID clients
# ----------------------------------------------------------------------

federated_clients = create_non_iid_clients(
    train_intensity,
    train_texture,
    train_spatial,
    train_deep,
    train_labels
)

print("\n" + "=" * 70)
print("CLIENT PARTITIONING READY FOR FEDERATED TRAINING")
print("=" * 70)