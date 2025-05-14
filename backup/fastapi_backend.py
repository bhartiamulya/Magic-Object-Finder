from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Selected classes and their emojis
CLASS_EMOJIS = {
    'apple': '🍎', 'aquarium_fish': '🐠', 'baby': '👶', 'bear': '🐻', 
    'beaver': '🦫', 'bed': '🛏️', 'bee': '🐝', 'beetle': '🪲',
    'bicycle': '🚲', 'bottle': '🍾', 'bowl': '🥣', 'boy': '👦', 
    'bridge': '🌉', 'bus': '🚌', 'butterfly': '🦋', 'camel': '🐪',
    'can': '🥫', 'castle': '🏰', 'caterpillar': '🐛', 'cattle': '🐄', 
    'chair': '🪑', 'chimpanzee': '🦧', 'clock': '⏰', 'cloud': '☁️',
    'cockroach': '🪳', 'couch': '🛋️', 'crab': '🦀', 'crocodile': '🐊', 
    'cup': '☕', 'dinosaur': '🦖', 'dolphin': '🐬', 'elephant': '🐘',
    'flatfish': '🐟', 'rose': '🌹', 'fox': '🦊', 'girl': '👧', 
    'hamster': '🐹', 'house': '🏠', 'kangaroo': '🦘', 'keyboard': '⌨️',
    'lamp': '💡', 'lawn_mower': '🚜', 'leopard': '🐆', 'lion': '🦁', 
    'lizard': '🦎', 'lobster': '🦞', 'man': '👨', 'maple_tree': '🍁',
    'mountain': '⛰️', 'mouse': '🐁'
}

SELECTED_CLASSES = list(CLASS_EMOJIS.keys())

# Load model once at startup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use ResNet50 to match the architecture used in training
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(SELECTED_CLASSES))
)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
model = model.to(device)

# Cache transform
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Cache no_grad context
no_grad_context = torch.no_grad()

@app.get("/")
async def root():
    return {
        "message": "Magic Object Finder API",
        "classes": [{"name": cls, "emoji": emoji} for cls, emoji in CLASS_EMOJIS.items()]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with no_grad_context:
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top 3 predictions
            top_prob, top_class = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                class_name = SELECTED_CLASSES[top_class[i].item()]
                confidence = top_prob[i].item()
                predictions.append({
                    "class": class_name,
                    "confidence": f"{confidence:.1%}",
                    "emoji": CLASS_EMOJIS[class_name],
                    "message": get_child_friendly_message(class_name, confidence)
                })
            
            return {
                "predictions": predictions,
                "message": "I found something exciting! 🎉"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Oops! Something went wrong: {str(e)}"
        )

def get_child_friendly_message(class_name, confidence):
    """Generate a child-friendly message based on confidence level"""
    if confidence > 0.9:
        return f"I'm super sure this is a {class_name}! {CLASS_EMOJIS[class_name]}"
    elif confidence > 0.7:
        return f"I think this looks like a {class_name}! {CLASS_EMOJIS[class_name]}"
    elif confidence > 0.5:
        return f"This might be a {class_name}... {CLASS_EMOJIS[class_name]}"
    else:
        return f"I'm not quite sure, but maybe it's a {class_name}? {CLASS_EMOJIS[class_name]}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
