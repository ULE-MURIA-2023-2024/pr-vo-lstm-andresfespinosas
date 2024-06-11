import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, num_layers)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])

# Load dataset
val_loader = DataLoader(
    VisualOdometryDataset(
        "dataset/val",
        transform=transform,
        sequence_length=sequence_length
    ),  # Agregar coma aquí
    batch_size=batch_size,
    shuffle=True
)

# val
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(torch.load("vo.pt"))
model.eval()

validation_string = ""
position = [0.0] * 7

with torch.no_grad():
    for images, labels, timestamp in tqdm(val_loader, desc="Validating:"):

        images = images.to(device)
        labels = labels.to(device)
        
        targets = model(images).cpu().numpy().tolist()
        
        for idx in range(len(timestamp)):
            target_str = ' '.join(map(str, targets[idx]))  # Convierte el target a una cadena sin corchetes
            validation_string += f"{timestamp[idx]} {target_str}"
            
            for pose in position:
                validation_string += f",{pose}"
                
            validation_string += "\n"

f = open("validation.txt", "a")
f.write(validation_string)
f.close()
