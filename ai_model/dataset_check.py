
import os

# ============================================
# MEMBER 1 - DATASET CHECK
# ============================================

DATASET_PATH = os.path.join("ai_model", "data")

TRAINING_PATH = os.path.join(DATASET_PATH, "Training")
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def count_images(folder_path):
    count = 0

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1

    return count


print("=" * 60)
print("        BRAIN TUMOR DATASET ANALYSIS")
print("=" * 60)

# ---------------- TRAINING ----------------

print("\nTRAINING DATASET")
print("-" * 40)

total_training = 0

for class_name in CLASSES:

    class_path = os.path.join(
        TRAINING_PATH,
        class_name
    )

    count = count_images(class_path)

    print(f"{class_name:15} : {count} images")

    total_training += count


print("-" * 40)
print(f"Total Training Images : {total_training}")


# ---------------- TESTING ----------------

print("\nTESTING DATASET")
print("-" * 40)

total_testing = 0

for class_name in CLASSES:

    class_path = os.path.join(
        TESTING_PATH,
        class_name
    )

    count = count_images(class_path)

    print(f"{class_name:15} : {count} images")

    total_testing += count


print("-" * 40)
print(f"Total Testing Images : {total_testing}")


# ---------------- TOTAL ----------------

print("\n" + "=" * 60)

total_images = total_training + total_testing

print(f"TOTAL DATASET IMAGES : {total_images}")

print("=" * 60)

print("\nDataset analysis completed successfully.")