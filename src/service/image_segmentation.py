import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional

from src.unet_config.models import configure_unet
from src.unet_config.models import device
import os


def segmentate_image(path: str):
    unet = configure_unet()
    image = Image.open(path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(image)
    width = tensor.shape[1]
    height = tensor.shape[2]
    alpha_channel = torch.ones(1, width, height)
    four_channel_tensor = torch.cat([tensor, alpha_channel], dim=0)
    resized_tensor = functional.resize(four_channel_tensor.unsqueeze(0), (384, 384))
    resized_tensor.to(device)
    segmentated_tensor = unet(resized_tensor)
    segmentated_images_dir = os.getenv("SEGMENTATED_IMAGES_DIR")
    image_file = path.split("/")[-1]
    segmentated_image_path = os.path.join(segmentated_images_dir, image_file)
    torchvision.utils.save_image(segmentated_tensor, segmentated_image_path)
    return segmentated_image_path
