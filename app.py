import gradio as gr
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import uuid
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for saving figures without GUI

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor()
])

# ResNet50 Feature Extractor
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(ResNetFeatureExtractor, self).__init__()
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Hook to extract feature maps.
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook) 

    def forward(self, input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]  # Feature map sizes h, w.
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)  # Merge the resized feature maps.
        patch = patch.reshape(patch.shape[1], -1).T  # Create a column tensor.

        return patch

# Initialize the model
backbone = ResNetFeatureExtractor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone.to(device)
memory_bank = None
best_threshold = None

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Function to load memory bank
def load_memory_bank(normal_images):
    global memory_bank, best_threshold
    
    memory_bank_list = []
    
    # Save images temporarily
    image_paths = []
    for i, img in enumerate(normal_images):
        img_path = f"uploads/normal_{i}.jpg"
        img.save(img_path)
        image_paths.append(img_path)
    
    # Collect features from normal images
    for pth in image_paths:
        with torch.no_grad():
            data = transform(Image.open(pth)).unsqueeze(0).to(device)
            features = backbone(data)
            memory_bank_list.append(features.cpu().detach())
    
    # Concatenate all features
    memory_bank = torch.cat(memory_bank_list, dim=0)
    
    # Only select 10% of total patches to avoid long inference time
    selected_indices = np.random.choice(len(memory_bank), size=max(len(memory_bank)//10, 1), replace=False)
    memory_bank = memory_bank[selected_indices]
    
    # Calculate threshold from normal data
    y_score = []
    for pth in image_paths:
        data = transform(Image.open(pth)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = backbone(data)
        distances = torch.cdist(features, memory_bank, p=2.0)
        dist_score, _ = torch.min(distances, dim=1) 
        s_star = torch.max(dist_score)
        y_score.append(s_star.cpu().numpy())
    
    # Set threshold as mean + 2*std
    best_threshold = np.mean(y_score) + 2 * np.std(y_score)
    
    return f"Memory bank created with {len(memory_bank)} feature vectors. Threshold: {best_threshold:.4f}"

# Function to detect anomalies
def detect_anomaly(test_image):
    global memory_bank, best_threshold
    
    if memory_bank is None:
        return [None, None, None, 0, "Please create memory bank first"]
    
    # Save image temporarily
    temp_path = "uploads/test_image.jpg"
    test_image.save(temp_path)
    
    with torch.no_grad():
        # Load and preprocess image
        test_tensor = transform(Image.open(temp_path)).unsqueeze(0).to(device)
        
        # Extract features
        features = backbone(test_tensor)
        
        # Calculate distances
        distances = torch.cdist(features, memory_bank, p=2.0)
        dist_score, _ = torch.min(distances, dim=1) 
        s_star = torch.max(dist_score)
        segm_map = dist_score.view(1, 1, 28, 28) 
        
        # Interpolate to original image size
        segm_map = torch.nn.functional.interpolate(
                    segm_map,
                    size=(224, 224),
                    mode='bilinear'
                ).cpu().squeeze().numpy()
        
        y_score_image = s_star.cpu().numpy()
        y_pred_image = 1 * (y_score_image >= best_threshold)
        class_label = ['OK', 'NOK']
        
        # Original image
        original_img = Image.open(temp_path)
        
        # Heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(segm_map, cmap='jet', vmin=best_threshold, vmax=best_threshold*2)
        plt.title(f'Anomaly Score: {y_score_image / best_threshold:.2f} ({class_label[y_pred_image]})')
        plt.axis('on')
        plt.tight_layout()
        heatmap_path = "results/heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()
        heatmap_img = Image.open(heatmap_path)
        
        # Segmentation map
        plt.figure(figsize=(5, 5))
        plt.imshow((segm_map > best_threshold*1.25), cmap='gray')
        plt.title('Anomaly Segmentation')
        plt.axis('on')
        plt.tight_layout()
        segmap_path = "results/segmap.png"
        plt.savefig(segmap_path)
        plt.close()
        segmap_img = Image.open(segmap_path)
        
        return [
            original_img, 
            heatmap_img, 
            segmap_img, 
            float(y_score_image / best_threshold), 
            class_label[y_pred_image]
        ]

# Create Gradio Interface
with gr.Blocks(title="ManuEncode: Autoencoder-Driven Anomaly Detection") as demo:
    gr.Markdown("# ManuEncode: Autoencoder-Driven Anomaly Detection for Precision Manufacturing Systems")
    
    with gr.Tab("Step 1: Create Memory Bank"):
        with gr.Row():
            normal_images_input = gr.Image(type="pil", label="Normal Images (Upload multiple)", sources=["upload"], interactive=True)
            normal_images_gallery = gr.Gallery(label="Selected Normal Images").style(grid=4, height="auto")
        
        normal_images_list = gr.State([])
        
        def add_to_gallery(image, images_list):
            if image is not None:
                images_list.append(image)
            return images_list, images_list
        
        normal_images_input.change(add_to_gallery, [normal_images_input, normal_images_list], [normal_images_list, normal_images_gallery])
        
        create_memory_bank_btn = gr.Button("Create Memory Bank")
        memory_bank_status = gr.Textbox(label="Status")
        
        def create_memory_bank_fn(images_list):
            if not images_list:
                return "Please upload at least one normal image"
            result = load_memory_bank(images_list)
            return result
        
        create_memory_bank_btn.click(create_memory_bank_fn, [normal_images_list], [memory_bank_status])
    
    with gr.Tab("Step 2: Detect Anomalies"):
        with gr.Row():
            test_image_input = gr.Image(type="pil", label="Test Image", sources=["upload"], interactive=True)
        
        detect_btn = gr.Button("Detect Anomalies")
        
        with gr.Row():
            original_output = gr.Image(label="Original Image")
            heatmap_output = gr.Image(label="Anomaly Heatmap")
            segmap_output = gr.Image(label="Anomaly Segmentation")
        
        with gr.Row():
            score_output = gr.Number(label="Anomaly Score")
            prediction_output = gr.Textbox(label="Prediction")
        
        detect_btn.click(
            detect_anomaly, 
            [test_image_input], 
            [original_output, heatmap_output, segmap_output, score_output, prediction_output]
        )

demo.launch()
