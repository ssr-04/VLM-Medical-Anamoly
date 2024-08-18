import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match ResNet input size
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pretrained model
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class_names = ['Diencephalon',
 'Frontal lobe',
 'Frontal lobe of lateral ventricle',
 'Parietal lobe',
 'Quadrigeminal cistern',
 'Temporal horn of lateral ventricle',
 'Third ventricle',
 'occipital lobe',
 'temporal lobe',
 'ventricles']  # Extract class labels

# Load your model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # Adjust final layer for your number of classes
model.load_state_dict(torch.load('./resnet50_brain_regions.pth'))  # Load the trained model weights
#model = model.to(device)

import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def predict_label(img_path="anomaly_heatmap_overlay.png"):

    img = Image.open(img_path)
    transform_inf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform_inf(img).reshape(1,3,224,224)
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = F.softmax(logits, dim=1)
    
    # Get the predicted class index
    _, predicted_class_index = torch.max(probabilities, 1)
    
    # Convert the index to the class label
    predicted_class_label = class_names[predicted_class_index.item()]
    query = ""
    # Print the results
    print(f"Predicted class: {predicted_class_label}")
    probs_class = list(probabilities.cpu().numpy()[0]*100)
    for c,percent in zip(class_names,probs_class):
        print(f"{c}: {percent:.2f}")
        query+=f"{c}: {percent:.2f}" + "\n"
    return query
