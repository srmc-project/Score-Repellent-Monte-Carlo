import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torchvision
from tqdm import tqdm
import pcd_ebm_ema
import vamp_utils
import mlp
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    else:
        print("Warning: Classifier checkpoint not found. Metrics will be random.")
        return None

def calculate_cumulative_metrics(cumulative_preds, batch_probs):
    """
    cumulative_preds: (N,)
    batch_probs: current (Batch, 10) - Vendi Score용
    """
    # 1. Cumulative KL Divergence
    counts = torch.bincount(cumulative_preds, minlength=10).float()
    empirical_dist = counts / counts.sum()
    
    target_dist = torch.ones_like(empirical_dist) / 10.0
    
    # KL Divergence 
    kl_div = F.kl_div((empirical_dist + 1e-10).log(), target_dist, reduction='sum').item()

    # 2. Batch Vendi Score 
    with torch.no_grad():
        norm_probs = F.normalize(batch_probs, p=2, dim=1)
        K = torch.mm(norm_probs, norm_probs.t()) / batch_probs.size(0)
        evals = torch.linalg.eigvalsh(K).real
        evals = evals[evals > 1e-10]
        vendi = torch.exp(-torch.sum(evals * torch.log(evals))).item()
        
    return kl_div, vendi, empirical_dist

# ==========================================
# Main Experiment 
# ==========================================
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def main(args):
    makedirs(args.save_dir)
    print(f"Experiment: Viz & Eval (Cumulative) in {args.save_dir}", flush=True)

    # --- 데이터 및 모델 로드 ---
    print("Loading dataset info...", flush=True)
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    
    if args.model.startswith("mlp-"):
        nint = int(args.model.split('-')[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    elif args.model.startswith("cnn-"):
        nint = int(args.model.split('-')[1])
        net = mlp.MNISTConvNet(nint)
    
    def preprocess(data): return torch.bernoulli(data) if args.dynamic_binarization else data
    
    # Init Mean 
    init_batch = []
    for x, _ in train_loader: init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps
    
    # Load model
    model = pcd_ebm_ema.EBM(net, init_mean)
    d = torch.load(args.ckpt_path, map_location=device)
    if args.ema: model.load_state_dict(d['ema_model'])
    else: model.load_state_dict(d['model'])
    model.to(device)
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    sampler = pcd_ebm_ema.get_sampler(args)
    classifier = get_classifier()

    # --- Initialization method ---
    if args.start_from == 'mean':
        print("Starting from: MEAN Image", flush=True)
        torch.manual_seed(args.seed)
        mean_img = init_mean.view(1, 28, 28).to(device)
        x_single = torch.bernoulli(mean_img).unsqueeze(0) 
        
    elif args.start_from == 'real':
        target_digit = args.start_digit
        print(f"Starting from: REAL Image (Target Digit: {target_digit})", flush=True)
        
        found = False
        
        print("Searching in test loader...", flush=True)
        try:
            for x_batch, y_batch in test_loader:
                matches = (y_batch == target_digit).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    idx = matches[0]
                    x_single = x_batch[idx].view(1, 1, 28, 28).to(device)
                    found = True
                    break
        except Exception as e:
            print(f"Test loader iteration skipped: {e}")
            
        if not found:
            print(f"Digit {target_digit} not found in loader. Fetching from torchvision...", flush=True)
            try:
                temp_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                                      transform=torchvision.transforms.ToTensor())
                for x, y in temp_mnist:
                    if y == target_digit:
                        if args.dataset_name == 'static_mnist':
                            x = (x > 0.5).float()
                        x_single = x.view(1, 1, 28, 28).to(device)
                        found = True
                        break
            except Exception as e:
                print(f"Error fetching from torchvision: {e}")

        if not found:
            raise ValueError(f"Digit {target_digit} could not be found anywhere!")

    x_sample = x_single.repeat(args.batch_size, 1, 1, 1).clone()
    
    torchvision.utils.save_image(x_sample, os.path.join(args.save_dir, "step_0000_start.png"), nrow=10)

    log_file = open(os.path.join(args.save_dir, "metrics_log.csv"), "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Step", "Cumulative_KL", "Batch_Vendi"])

    cumulative_preds_list = []

    print(f"Running {args.total_steps} steps (Burn-in: {args.burn_in})...", flush=True)
    
    for step in tqdm(range(0, args.total_steps + 1)):
        is_burning = step < args.burn_in

        if not is_burning:
            if step % args.eval_every == 0:
                if classifier:
                    with torch.no_grad():
                        normalized_batch = (x_sample - 0.1307) / 0.3081
                        logits = classifier(normalized_batch)
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        
                        cumulative_preds_list.append(preds)
                        all_preds = torch.cat(cumulative_preds_list)
                        
                        kl, vendi, dist_probs = calculate_cumulative_metrics(all_preds, probs)
                        
                        writer.writerow([step, f"{kl:.4f}", f"{vendi:.4f}"])
                        log_file.flush()
                        
                        if step % (args.eval_every * 5) == 0:
                             dist_str = " | ".join([f"{p:.2f}" for p in dist_probs])
                             tqdm.write(f"\n[Step {step}] Cumul KL: {kl:.4f} | Batch Vendi: {vendi:.2f}")
                             tqdm.write(f"Dist: [{dist_str}]")

        if step % args.viz_every == 0:
            save_name = f"step_{step:04d}.png"
            if is_burning:
                save_name = f"burnin_{step:04d}.png" 
            
            torchvision.utils.save_image(
                x_sample, 
                os.path.join(args.save_dir, save_name), 
                normalize=True, 
                nrow=10
            )
            
        # sampling start
        if step < args.total_steps:
            x_flat = x_sample.view(x_sample.size(0), -1)
            x_next_flat = sampler.step(x_flat.detach(), model).detach()
            x_sample = x_next_flat.view(x_sample.size(0), 1, 28, 28)

    log_file.close()
    print("Done! Metrics saved to metrics_log.csv", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    
    parser.add_argument('--start_from', type=str, default='real', choices=['mean', 'real'])
    parser.add_argument('--start_digit', type=int, default=7)
    
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--burn_in', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    
    parser.add_argument('--sampler', type=str, default='gwg')
    parser.add_argument('--alpha', type=float, default=1.0)
    
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--model', type=str, default='resnet-64')
    parser.add_argument('--base_dist', action='store_true', default=True)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--n_out', type=int, default=3)
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--steps_per_iter', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=100)
    
    args = parser.parse_args()
    main(args)