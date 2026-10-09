import os
import matplotlib.pyplot as plt

# Dataset paths
train_path = os.path.join("ai_model", "data", "Training")
test_path = os.path.join("ai_model", "data", "Testing")

# Classes
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Count images in each class
train_counts = []
test_counts = []

for class_name in classes:
    train_folder = os.path.join(train_path, class_name)
    test_folder = os.path.join(test_path, class_name)

    train_count = len([
        f for f in os.listdir(train_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    test_count = len([
        f for f in os.listdir(test_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    train_counts.append(train_count)
    test_counts.append(test_count)

# Display counts
print("\nDATASET VISUALIZATION")
print("-" * 40)

for i, class_name in enumerate(classes):
    print(f"{class_name}:")
    print(f"  Training: {train_counts[i]} images")
    print(f"  Testing : {test_counts[i]} images")

print("-" * 40)
print(f"Total Training Images: {sum(train_counts)}")
print(f"Total Testing Images : {sum(test_counts)}")
print(f"Total Dataset Images : {sum(train_counts) + sum(test_counts)}")

# Create visualization
x = range(len(classes))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.2 for i in x],
    train_counts,
    width=0.4,
    label="Training"
)

plt.bar(
    [i + 0.2 for i in x],
    test_counts,
    width=0.4,
    label="Testing"
)

plt.xlabel("Brain Tumor Classes")
plt.ylabel("Number of Images")
plt.title("Training and Testing Dataset Distribution")
plt.xticks(list(x), classes)
plt.legend()

plt.tight_layout()

# Save graph
output_file = os.path.join("ai_model", "dataset_distribution.png")
plt.savefig(output_file)

print("\nVisualization saved successfully.")
print(f"Saved as: {output_file}")

plt.show()