import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os


class BrainTumorClassifier:
    def __init__(self, model_path="model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the traced model
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the same transforms used during training/testing
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Class labels
        self.labels = [
                        "Glioma Tumor",
                        "Meningioma Tumor",
                        "No Tumor",
                        "Pituitary Tumor",
                    ]

    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None

        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Create prediction dictionary
        return {self.labels[idx]: float(prob) for idx, prob in enumerate(probabilities)}


# Create classifier instance
classifier = BrainTumorClassifier()

# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=4),
    title="Brain Tumor Classifier",
    description="Upload an image to classify the type of brain tumor",
    examples=[
        ["examples/glioma.jpg"], 
        ["examples/meningioma.jpg"], 
        ["examples/no_tumor.jpg"], 
        ["examples/pituitary.jpg"]],
)

if __name__ == "__main__":
    demo.launch()
