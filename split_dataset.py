import os
import shutil
import random

SOURCE = r"C:\Users\kiran\OneDrive\Desktop\plant_images"
DEST = r"C:\Users\kiran\OneDrive\Desktop\CNN\data"

random.seed(42)

categories = [
    "chili_baby",
    "chili_mature",
    "potato_baby",
    "potato_mature",
    "tomato_baby",
    "tomato_mature"
]

for category in categories:

    if "_" in category:
        plant, stage = category.split("_")

    source_folder = os.path.join(SOURCE, category)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    test_images = images[split_index:]

    train_folder = os.path.join(DEST, "train", plant, stage)
    test_folder = os.path.join(DEST, "test", plant, stage)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(test_folder, image)
        )

    print(
        category,
        "→ Train:", len(train_images),
        "| Test:", len(test_images)
    )

print("\nDataset split completed successfully!")