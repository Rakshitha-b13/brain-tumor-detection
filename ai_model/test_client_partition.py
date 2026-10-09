import os
import numpy as np
from collections import Counter

# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42
NUM_CLIENTS = 3
NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_RESULTS_DIR = os.path.join(BASE_DIR, "fusion_results")

np.random.seed(SEED)


# ================================================================
# LOAD PCA-REDUCED TRAINING FEATURES
# ================================================================

print("=" * 70)
print("LOADING PCA-REDUCED TRAINING FEATURES")
print("=" * 70)

train_data = np.load(
    os.path.join(
        FUSION_RESULTS_DIR,
        "train_features_pca.npz"
    )
)

intensity_features = train_data["intensity"]
texture_features = train_data["texture"]
spatial_features = train_data["spatial"]
deep_features = train_data["deep"]
labels = train_data["labels"]

print(f"Intensity : {intensity_features.shape}")
print(f"Texture   : {texture_features.shape}")
print(f"Spatial   : {spatial_features.shape}")
print(f"Deep      : {deep_features.shape}")
print(f"Labels    : {labels.shape}")


# ================================================================
# CREATE NON-IID CLIENT PARTITIONS
# ================================================================

print("\n" + "=" * 70)
print("CREATING NON-IID CLIENT PARTITIONS")
print("=" * 70)

rng = np.random.default_rng(SEED)

# Get indices belonging to each class
class_indices = {
    class_id: np.where(labels == class_id)[0]
    for class_id in range(NUM_CLASSES)
}

# Shuffle each class independently
for class_id in class_indices:
    rng.shuffle(class_indices[class_id])


# ================================================================
# CLIENT CLASS DISTRIBUTIONS
# ================================================================
#
# Total training samples = 5600
#
# Client 1 = 1400
# Client 2 = 2000
# Client 3 = 2200
#
# The distributions are intentionally heterogeneous.
# ================================================================

client_class_counts = {
    0: {0: 800, 1: 400, 2: 100, 3: 100},
    1: {0: 300, 1: 700, 2: 300, 3: 700},
    2: {0: 300, 1: 300, 2: 1000, 3: 600}
}


client_indices = {}

for client_id in range(NUM_CLIENTS):

    selected_indices = []

    for class_id in range(NUM_CLASSES):

        count = client_class_counts[client_id][class_id]

        selected = class_indices[class_id][:count]

        class_indices[class_id] = class_indices[class_id][count:]

        selected_indices.extend(selected.tolist())

    rng.shuffle(selected_indices)

    client_indices[client_id] = np.array(
        selected_indices,
        dtype=np.int32
    )


# ================================================================
# VERIFY PARTITION
# ================================================================

print("\n" + "=" * 70)
print("VERIFYING CLIENT PARTITION")
print("=" * 70)

all_indices = np.concatenate(
    [client_indices[i] for i in range(NUM_CLIENTS)]
)

assert len(all_indices) == len(labels)

assert len(np.unique(all_indices)) == len(labels)

assert set(all_indices.tolist()) == set(
    range(len(labels))
)

print("Total samples assigned :", len(all_indices))
print("Unique samples         :", len(np.unique(all_indices)))
print("Duplicate samples      : 0")
print("Missing samples        : 0")

print("\nPartition verification PASSED.")


# ================================================================
# DISPLAY CLIENT DISTRIBUTIONS
# ================================================================

print("\n" + "=" * 70)
print("NON-IID CLIENT DISTRIBUTIONS")
print("=" * 70)

for client_id in range(NUM_CLIENTS):

    idx = client_indices[client_id]

    client_labels = labels[idx]

    counts = Counter(client_labels.tolist())

    print(f"\nCLIENT {client_id + 1}")
    print("-" * 40)

    for class_id, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:<15}: "
            f"{counts.get(class_id, 0):>4}"
        )

    print("-" * 40)

    print(
        f"{'Total':<15}: "
        f"{len(idx):>4}"
    )


# ================================================================
# SAVE CLIENT INDICES
# ================================================================

client_index_path = os.path.join(
    FUSION_RESULTS_DIR,
    "non_iid_client_indices.npz"
)

np.savez(
    client_index_path,
    client_1=client_indices[0],
    client_2=client_indices[1],
    client_3=client_indices[2]
)

print("\n" + "=" * 70)
print("NON-IID PARTITIONING COMPLETED")
print("=" * 70)

print("\nClient index file saved to:")
print(client_index_path)