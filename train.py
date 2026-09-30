

import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import PlantDataset, PlantNet, train_transform, eval_transform


DATA_TRAIN_DIR = "data/train"
DATA_TEST_DIR = "data/test"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoint.pth"


def main():
    print(f"Using device: {DEVICE}")

    train_ds = PlantDataset(DATA_TRAIN_DIR, transform=train_transform)
    test_ds = PlantDataset(
        DATA_TEST_DIR,
        transform=eval_transform,
        species_classes=train_ds.species_classes,  # keep same label order
        stage_classes=train_ds.stage_classes,
    )

    print("Plant classes found:", train_ds.species_classes)
    print("Stage classes found:", train_ds.stage_classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = PlantNet(
        num_species=len(train_ds.species_classes),
        num_stages=len(train_ds.stage_classes),
    ).to(DEVICE)

    species_criterion = nn.CrossEntropyLoss()
    stage_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for images, species_labels, stage_labels in train_loader:
            images = images.to(DEVICE)
            species_labels = species_labels.to(DEVICE)
            stage_labels = stage_labels.to(DEVICE)

            optimizer.zero_grad()
            species_logits, stage_logits = model(images)

            loss_species = species_criterion(species_logits, species_labels)
            loss_stage = stage_criterion(stage_logits, stage_labels)
            loss = loss_species + loss_stage  # combined multi-task loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # ---- Evaluate ----
        model.eval()
        correct_species, correct_stage, total = 0, 0, 0
        with torch.no_grad():
            for images, species_labels, stage_labels in test_loader:
                images = images.to(DEVICE)
                species_labels = species_labels.to(DEVICE)
                stage_labels = stage_labels.to(DEVICE)

                species_logits, stage_logits = model(images)
                species_pred = species_logits.argmax(dim=1)
                stage_pred = stage_logits.argmax(dim=1)

                correct_species += (species_pred == species_labels).sum().item()
                correct_stage += (stage_pred == stage_labels).sum().item()
                total += images.size(0)

        species_acc = correct_species / total
        stage_acc = correct_stage / total

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"species_acc={species_acc:.3f} | "
              f"stage_acc={stage_acc:.3f}")

        # Save best model (by species accuracy, the main task)
        if species_acc > best_acc:
            best_acc = species_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "species_classes": train_ds.species_classes,
                "stage_classes": train_ds.stage_classes,
            }, CHECKPOINT_PATH)
            print(f"  -> Saved new best checkpoint ({CHECKPOINT_PATH})")

    print("Training complete. Best species accuracy:", best_acc)


if __name__ == "__main__":
    main()
