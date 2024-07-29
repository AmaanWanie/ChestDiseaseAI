import streamlit as st
import torchvision
import torch
import numpy as np
from PIL import Image

st.title("COVID-19 detection using resnet")

img = st.file_uploader("Upload the image", type=['png', 'jpg', 'jpeg'])

if img:
    st.image(img,width=300)

    def model():
        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)  # Output layer for 3 classes
        resnet18.load_state_dict(torch.load("resnet18.pth"))
        resnet18.eval()

        try:
            # Convert image to PyTorch tensor with correct shape
            image = Image.open(img)
            image = image.convert("RGB")  # Ensure 3 channels
            image = image.resize((224, 224))  # Resize for ResNet18
            image = np.array(image) / 255.0  # Normalize pixel values
            image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Normalize image using pre-trained model's mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            image = (image - mean) / std

            # Inference
            with torch.no_grad():
                outputs = resnet18(image)
                _, predicted = torch.max(outputs, 1)
            return predicted.item()
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

    if img:
        prediction = model()
        if prediction is not None:
            class_labels = ["Normal", "Viral", "COVID"]
            st.write("Predicted class:", class_labels[prediction])

    
