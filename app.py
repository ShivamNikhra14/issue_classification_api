from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import requests

# ---------------------------
# Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
num_classes = len(class_names)

# ---------------------------
# Model Definition
# ---------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE[0] // 8) * (IMG_SIZE[1] // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Load Model
# ---------------------------
model = CNNModel(num_classes)
model.load_state_dict(torch.load("model/urban_issue_cnn.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------------------
# Transform for incoming image
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Urban Issue Classification API")

@app.get("/")
def root():
    return {"message": "Urban Issue CNN API is running!"}

# ---------------------------
# Updated /predict endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        # 1️⃣ Get image bytes from file or URL
        if file:
            image_bytes = await file.read()
        elif image_url:
            response = requests.get(image_url)
            if response.status_code != 200:
                return JSONResponse({"error": "Failed to download image from URL"}, status_code=400)
            image_bytes = response.content
        else:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        # 2️⃣ Process image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        # 3️⃣ Run through model
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        return JSONResponse({"predicted_class": label})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
