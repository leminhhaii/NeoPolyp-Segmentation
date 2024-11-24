import argparse
import torch
import cv2
import segmentation_models_pytorch as smp
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


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

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

input_image = cv2.imread(args.image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
ori_w = input_image.shape[0]
ori_h = input_image.shape[1]
img = cv2.resize(input_image, (256, 256))
transformed = val_transform(image=img)
input_img = transformed["image"]
input_img = input_img.unsqueeze(0).to(device)

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output) 

with torch.inference_mode():
    output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
mask = cv2.resize(output_mask, (ori_h, ori_w))
mask = np.argmax(mask, axis=2)
mask_rgb = mask_to_rgb(mask, color_dict)
mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("segmentation_result.png", mask_rgb) 

print("Done")
