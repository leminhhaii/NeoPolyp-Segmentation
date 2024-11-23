import argparse
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np

parser = argparse.ArgumentParser(description="Run inference on an image using a pretrained segmentation model.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
args = parser.parse_args()

model = smp.UnetPlusPlus(
    encoder_name="resnet34",      
    encoder_weights='imagenet',          
    in_channels=3,                 
    classes=3                     
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("model.pth", map_location=device)  
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])  
else:
    model.load_state_dict(checkpoint)  

model.to(device)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize((512, 512)),   
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

input_image = Image.open(args.image_path).convert("RGB")  
input_tensor = preprocess(input_image).unsqueeze(0).to(device)

with torch.inference_mode():
    output = model(input_tensor)
    output = torch.argmax(output, dim=1).squeeze().cpu().numpy()

colormap = {
    0: (0, 0, 0),       
    1: (0, 255, 0),     
    2: (255, 0, 0),     
}

def apply_colormap(segmentation_output):
    h, w = segmentation_output.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        color_image[segmentation_output == class_id] = color

    return Image.fromarray(color_image)

segmented_color_image = apply_colormap(output)
segmented_color_image.save("segmented_image_color.png")
print("Segmented image with color saved as segmented_image_color.png")
