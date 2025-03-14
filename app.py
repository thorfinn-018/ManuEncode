from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
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

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "anomaly_detection_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

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

# Initialize the model and memory bank
backbone = ResNetFeatureExtractor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone.to(device)
memory_bank = None
best_threshold = None

# Function to load memory bank
def load_memory_bank(normal_image_folder):
    global memory_bank, best_threshold
    
    memory_bank_list = []
    
    folder_path = Path(normal_image_folder)
    
    # Collect features from normal images
    for pth in folder_path.iterdir():
        if pth.suffix.lower() in ['.jpg', '.jpeg', '.png']:
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
    for pth in folder_path.iterdir():
        if pth.suffix.lower() in ['.jpg', '.jpeg', '.png']:
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
def detect_anomaly(image_path, save_dir, filename_base):
    with torch.no_grad():
        # Load and preprocess image
        test_image = transform(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Extract features
        features = backbone(test_image)
        
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
        
        # Create and save visualizations
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.figure(figsize=(5, 5))
        plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.title('Original Image')
        plt.axis('on')
        plt.tight_layout()
        original_path = os.path.join(save_dir, f"{filename_base}_original.png")
        plt.savefig(original_path)
        
        # Heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(segm_map, cmap='jet', vmin=best_threshold, vmax=best_threshold*2)
        plt.title(f'Anomaly Score: {y_score_image / best_threshold:.2f} ({class_label[y_pred_image]})')
        plt.axis('on')
        plt.tight_layout()
        heatmap_path = os.path.join(save_dir, f"{filename_base}_heatmap.png")
        plt.savefig(heatmap_path)
        
        # Ground truth / Segmentation map
        plt.figure(figsize=(5, 5))
        plt.imshow((segm_map > best_threshold*1.25), cmap='gray')
        plt.title('Anomaly Segmentation')
        plt.axis('on')
        plt.tight_layout()
        segmap_path = os.path.join(save_dir, f"{filename_base}_segmap.png")
        plt.savefig(segmap_path)
        
        plt.close('all')
        
        result = {
            'original': original_path,
            'heatmap': heatmap_path,
            'segmap': segmap_path,
            'score': float(y_score_image / best_threshold),
            'prediction': class_label[y_pred_image]
        }
        
        return result

# Routes
@app.route('/')
def index():
    return render_template('index.html', memory_bank_loaded=(memory_bank is not None))

@app.route('/upload_memory_bank', methods=['GET', 'POST'])
def upload_memory_bank():
    if request.method == 'POST':
        if 'normal_images' not in request.files:
            flash('No files selected')
            return redirect(request.url)
            
        files = request.files.getlist('normal_images')
        
        if not files or files[0].filename == '':
            flash('No files selected')
            return redirect(request.url)
            
        # Create a temporary directory for normal images
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'normal_images')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded normal images
        for file in files:
            if file and file.filename:
                file.save(os.path.join(temp_dir, file.filename))
        
        # Load memory bank
        message = load_memory_bank(temp_dir)
        flash(message)
        
        return redirect(url_for('index'))
        
    return render_template('upload_memory_bank.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if memory_bank is None:
        flash('Please upload normal images to create a memory bank first')
        return redirect(url_for('upload_memory_bank'))
        
    if request.method == 'POST':
        if 'test_image' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['test_image']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        if file:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            filename_base = os.path.splitext(filename)[0]
            result = detect_anomaly(filepath, app.config['RESULTS_FOLDER'], filename_base)
            
            return render_template('result.html', result=result)
    
    return render_template('detect.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)