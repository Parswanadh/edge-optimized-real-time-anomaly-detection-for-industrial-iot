import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import EdgeModel  # Assuming you have a custom model file named models.py
from datasets import IndustrialDataset  # Assuming you have a custom dataset file named datasets.py
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model from checkpoint
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Define the evaluation function
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)

# Load the dataset and create a DataLoader
test_dataset = IndustrialDataset('path/to/test/data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluation setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('path/to/checkpoint.pth')
model.to(device)

# Perform evaluation
preds, labels = evaluate(model, test_loader)

# Calculate evaluation metrics
accuracy = (preds == labels).float().mean()
precision = None  # Implement precision calculation if needed
recall = None    # Implement recall calculation if needed
f1_score = None  # Implement F1-Score calculation if needed

# Visualize results
def visualize_results(inputs, preds, labels):
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        input_img = inputs[i].cpu().numpy().transpose((1, 2, 0))
        input_img = np.clip(input_img, 0, 1)
        ax.imshow(input_img)
        ax.set_title(f"Pred: {preds[i]}, Label: {labels[i]}")
    plt.show()

# Visualize a sample of the test data
if len(test_dataset) > 0:
    visualize_results(test_dataset[:64][0], preds, labels)

# Performance reporting
print(f"Accuracy: {accuracy:.4f}")
if precision is not None and recall is not None:
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")