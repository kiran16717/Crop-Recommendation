import json
import subprocess
import os

categories = {
    "tomato_baby": r"C:\Users\kiran\OneDrive\Desktop\plant_images\tomato_baby",
    "tomato_mature": r"C:\Users\kiran\OneDrive\Desktop\plant_images\tomato_mature",
    "potato_baby": r"C:\Users\kiran\OneDrive\Desktop\plant_images\potato_baby",
    "potato_mature": r"C:\Users\kiran\OneDrive\Desktop\plant_images\potato_mature",
    "chili_baby": r"C:\Users\kiran\OneDrive\Desktop\plant_images\chili_baby",
    "chili_mature": r"C:\Users\kiran\OneDrive\Desktop\plant_images\chili_mature"
}

output_folder = r"C:\Users\kiran\OneDrive\Desktop\CNN\json_reports"
os.makedirs(output_folder, exist_ok=True)

for category, folder in categories.items():

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    images.sort()

    if not images:
        print("No images found:", category)
        continue

    print(f"\nProcessing {category} ({len(images)} images)...")

    for image_name in images:

        image_path = os.path.join(folder, image_name)

        result = subprocess.run(
            ["python", "predict.py", image_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Prediction failed:", image_name)
            print(result.stderr)
            continue

        try:
            prediction = json.loads(result.stdout)
        except json.JSONDecodeError:
            print("Invalid prediction output:", image_name)
            continue

        # One JSON file for each image
        report = {
            "dataset_category": category,
            "image": image_name,
            "plant": prediction["plant_name"],
            "growth_stage": prediction["growth_stage"],
            "confidence": prediction["confidence_score"],
            "stage_confidence": prediction["stage_confidence_score"],
            "model": "CNN"
        }

        # Remove extension from image name
        image_base_name = os.path.splitext(image_name)[0]

        output_file = os.path.join(
            output_folder,
            f"{category}_{image_base_name}.json"
        )

        with open(output_file, "w") as file:
            json.dump(report, file, indent=4)

        print("Created:", output_file)

print("\nAll image-wise JSON reports created successfully!")