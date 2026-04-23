import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from glob import glob
from PIL import Image
import numpy as np
from torchvision import transforms

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Simple Classifier (Same as before)
# ==========================================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def get_classifier(ckpt_path="mnist_classifier.pt"):
    model = SimpleClassifier().to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return model
    print("Classifier checkpoint not found! Please run the previous training code first.")
    return None

# ==========================================
# 2. Metric Calculation (Entropy, KL, Vendi)
# ==========================================
def calculate_metrics(image_folder, classifier):
    image_paths = sorted(glob(os.path.join(image_folder, "*.png")))
    if len(image_paths) == 0:
        print(f"No images found in {image_folder}")
        return None, None, None

    print(f"Loading {len(image_paths)} images...")
    
    images = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    for p in image_paths:
        img = Image.open(p).convert('L')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    # Process in batches to avoid OOM
    batch_size = 100
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size]).to(device)
            logits = classifier(batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
            
    all_probs = torch.cat(all_probs, dim=0) # (N, 10)
    
    # ---------------------------------------------------------
    # Metric 1: Class Entropy (Higher is Better)
    # ---------------------------------------------------------
    # Average distribution across the dataset
    marginal_dist = all_probs.mean(dim=0) 
    entropy = -torch.sum(marginal_dist * torch.log(marginal_dist + 1e-8)).item()
    
    # ---------------------------------------------------------
    # Metric 2: KL Divergence (Lower is Better)
    # ---------------------------------------------------------
    # How different is the marginal distribution from a Uniform distribution?
    target_dist = torch.ones_like(marginal_dist) / 10.0
    # KL(P || Q) = sum(p * log(p/q))
    # Note: F.kl_div expects log_prob as input
    kl_div = F.kl_div((marginal_dist + 1e-10).log(), target_dist, reduction='sum').item()
    
    # ---------------------------------------------------------
    # Metric 3: Vendi Score (Higher is Better)
    # ---------------------------------------------------------
    # Calculates diversity based on the similarity of prediction probabilities
    # K(x, y) = <p(x), p(y)>
    norm_probs = F.normalize(all_probs, p=2, dim=1)
    # Computationally expensive for large N, subsample if N > 2000
    if norm_probs.size(0) > 2000:
        idx = torch.randperm(norm_probs.size(0))[:2000]
        norm_probs = norm_probs[idx]
        
    K = torch.mm(norm_probs, norm_probs.t()) / norm_probs.size(0)
    evals = torch.linalg.eigvalsh(K).real
    evals = evals[evals > 1e-10]
    vendi = torch.exp(-torch.sum(evals * torch.log(evals))).item()
    
    return entropy, kl_div, vendi

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    args = parser.parse_args()
    
    classifier = get_classifier()
    if classifier:
        entropy, kl, vendi = calculate_metrics(args.image_dir, classifier)
        
        if entropy is not None:
            print("\n" + "="*40)
            print(f" Folder: {args.image_dir}")
            print(f" 1. Class Entropy (↑) : {entropy:.4f} (Max ~2.30)")
            print(f" 2. KL Divergence (↓) : {kl:.4f} (0 is Perfect Mixing)")
            print(f" 3. Vendi Score   (↑) : {vendi:.4f} (Max 10.0)")
            print("="*40 + "\n")