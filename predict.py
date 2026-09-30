"""
predict.py
----------
Loads the trained checkpoint and predicts, for a single image:
    - plant name
    - plant id (class index)
    - confidence score (species)
    - growth stage (baby / mature)
    - stage confidence score

Run from VS Code terminal:
    python predict.py path/to/image.jpg
"""

import sys
import json
import torch
from PIL import Image

from model import PlantNet, eval_transform

CHECKPOINT_PATH = "checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    species_classes = checkpoint["species_classes"]
    stage_classes = checkpoint["stage_classes"]

    model = PlantNet(
        num_species=len(species_classes),
        num_stages=len(stage_classes),
        pretrained=False,  # weights come from checkpoint, not ImageNet
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, species_classes, stage_classes


def predict_image(image_path):
    model, species_classes, stage_classes = load_model()

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        species_logits, stage_logits = model(tensor)
        species_probs = torch.softmax(species_logits, dim=1)[0]
        stage_probs = torch.softmax(stage_logits, dim=1)[0]

        species_idx = int(species_probs.argmax().item())
        stage_idx = int(stage_probs.argmax().item())

        result = {
            "plant_name": species_classes[species_idx],
            "plant_id": species_idx,
            "confidence_score": round(float(species_probs[species_idx]), 4),
            "growth_stage": stage_classes[stage_idx],
            "stage_confidence_score": round(float(stage_probs[stage_idx]), 4),
        }
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    output = predict_image(image_path)
    print(json.dumps(output, indent=2))
