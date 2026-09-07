import kagglehub
import os

print("Downloading KCC dataset from Kaggle...")
path = kagglehub.dataset_download("sridhargutam/kcc-dataset")
print("Path to dataset files:", path)

print("\nFiles in the dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
