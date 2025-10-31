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
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else None # The image to predict is provided as a CLI argument: python predict_one.py path/to/image.jpg
TOP_K = 3  # NEW show top-3 predictions
CONFIDENCE_THRESHOLD = 0.1  # NEW do not show predictions below 10%

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
        raise FileNotFoundError(f"No .pt model in {MODEL_PATH}/")
    
    model_file = pts[-1]
    print(f"📦 Loading model: {model_file}")
    
    ckpt = torch.load(os.path.join(MODEL_PATH, model_file), map_location="cpu")  # Loading the checkpoint (with the model weight and a list of classes)
    classes = ckpt["classes"]
    # Rebuild same model architecture and load weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

# NEW show metrics from checkpoint
    if "val_acc" in ckpt:
        print(f"   Model accuracy: {ckpt['val_acc']:.1%} (val), {ckpt.get('train_acc', 0):.1%} (train)")
    if "epoch" in ckpt:
        print(f"   Trained for: {ckpt['epoch']} epochs")

    return model, classes

def predict_image(image_path: str, model, classes):
    """NEW predict an image class"""
    # Modified: Added ImageNet normalization (like in training!)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1).squeeze().tolist()
    
    return list(zip(classes, probs))


# -------------------- MAIN INFERENCE --------------------

def main():
    if not IMG_PATH:
        print("Usage: python src/ml/predict_one.py /path/to/image.jpg")
        sys.exit(1)
    
    if not os.path.exists(IMG_PATH):
        print(f"❌ Error: Image not found: {IMG_PATH}")
        sys.exit(1)

    # Load model
    model, classes = load_latest_model()
    # Predict
    print(f"\n🔮 Predicting: {IMG_PATH}")
    results = predict_image(IMG_PATH, model, classes)
    
    # Sort by probability
    results = sorted(results, key=lambda t: t[1], reverse=True)
    
    # NEW showing top-K with confidence
    print(f"\n📊 Top {TOP_K} predictions:")
    print("-" * 40)

    for i, (cls, prob) in enumerate(results[:TOP_K], 1):
        if prob < CONFIDENCE_THRESHOLD:
            continue
        
        # Visual indicator of confidence
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"{i}. {cls:<15} {prob:>6.1%} {bar}")
    
    # NEW Show if the model is very uncertain
    top1_prob = results[0][1]
    if top1_prob < 0.5:
        print(f"\n⚠️  Low confidence ({top1_prob:.1%}) - model is uncertain!")
        print("   Consider: better image quality, different angle, or unseen category")
    
    print("-" * 40)
    print(f"\n✅ Best guess: {results[0][0]} ({results[0][1]:.1%})")

# -------------------- ENTRYPOINT --------------------

if __name__ == "__main__":
    main()
