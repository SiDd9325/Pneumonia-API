from fastapi import FastAPI, UploadFile, File
from PIL import Image
from model.pneumonia_model import get_model
from utils.preprocess import preprocess_image

app = FastAPI()
model = get_model("weights/pneumonia_model.pth")
@app.get("/")
def read_root():
    return {"message": "Welcome to the Pneumonia Detection API"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    tensor = preprocess_image(image)
    prediction, confidence = model.predict(tensor)
    label = "Pneumonia" if prediction == 1 else "Normal"
    return {"prediction": label, "confidence": round(confidence, 4)}
