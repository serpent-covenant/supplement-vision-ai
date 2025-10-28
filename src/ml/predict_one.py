#Run inference on a single image using the latest saved ResNet-18 checkpoint.
#    - Automatically loads the newest .pt model from MODELS_DIR
#    - Applies the same preprocessing as during training (Resize + ToTensor)
#    - Prints class probabilities sorted in descending order


import os, sys, torch
from PIL import Image
from torchvision import transforms, models
from torch import nn

# -------------------- PATHS --------------------

MODEL_PATH = "models"  # Directory containing saved .pt checkpoints
IMG_PATH = sys.argv[1]  #   The image to predict is provided as a CLI argument: python predict_one.py path/to/image.jpg

# -------------------- LOAD MODEL --------------------

def load_latest_model():
        # EN: Load the most recently saved model checkpoint (.pt) from MODEL_PATH.
        #
        #    Returns
        #    -------
        #    model   : torch.nn.Module — loaded and ready for inference
        #    classes : list[str]       — list of class names from the checkpoint
    pts = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith(".pt")]) #Find all .pt files and sort them lexicographically (latest is last)
    if not pts:
        raise FileNotFoundError("No .pt model in models/")
    ckpt = torch.load(os.path.join(MODEL_PATH, pts[-1]), map_location="cpu")  # Loading the checkpoint (with the model weight and a list of classes)
    classes = ckpt["classes"]
    # Rebuild same model architecture and load weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes

# -------------------- MAIN INFERENCE --------------------

def main():
        # Run prediction for one image:
        #   - Load model
        #   - Apply transforms (resize, tensor)
        #   - Get softmax probabilities
        #   - Print sorted results
     
    model, classes = load_latest_model()
    # Define same preprocessing used during training
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # Open image with PIL (ensure RGB mode)
    img = Image.open(IMG_PATH).convert("RGB")
    # Apply transform and add batch dimension (unsqueeze -> [1, C, H, W])
    x = tfm(img).unsqueeze(0)
    with torch.no_grad(): # Gradient-free inference
        logits = model(x)
        probs = logits.softmax(dim=1).squeeze().tolist()
    ranked = sorted(zip(classes, probs), key=lambda t: t[1], reverse=True) # Combining class names and probabilities
    for cls, p in ranked:
        print(f"{cls}: {p:.3f}")

# -------------------- ENTRYPOINT --------------------

if __name__ == "__main__":
    # Verify that an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python src/ml/predict_one.py /path/to/image.jpg")
        sys.exit(1)
    main()
