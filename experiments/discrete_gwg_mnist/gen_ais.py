import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
from tqdm import tqdm
import pcd_ebm_ema  
import vamp_utils
import mlp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AISModel(nn.Module):
    def __init__(self, model, init_dist):
        super().__init__()
        self.model = model
        self.init_dist = init_dist

    def forward(self, x, beta):
        logpx = self.model(x).squeeze()
        logpi = self.init_dist.log_prob(x).sum(-1)
        return logpx * beta + logpi * (1. - beta)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def main(args):

    makedirs(args.save_dir)
    print(f"Saving AIS samples to: {args.save_dir}")

    print("Loading dataset info...")
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps
    
    print("Loading Model...")
    if args.model.startswith("mlp-"):
        nint = int(args.model.split('-')[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    elif args.model.startswith("cnn-"):
        nint = int(args.model.split('-')[1])
        net = mlp.MNISTConvNet(nint)
    else:
        raise ValueError("invalid model definition")

    model = pcd_ebm_ema.EBM(net, init_mean)
    
    print(f"Loading checkpoint from: {args.ckpt_path}")
    d = torch.load(args.ckpt_path, map_location=device)
    if args.ema:
        model.load_state_dict(d['ema_model'])
    else:
        model.load_state_dict(d['model'])
    
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    sampler = pcd_ebm_ema.get_sampler(args)
    print(f"Sampler: {sampler}")

    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(device))
    
    ais_model = AISModel(model, init_dist)

    total_samples = args.n_samples
    batch_size = args.batch_size
    num_batches = int(np.ceil(total_samples / batch_size))
    
    betas = np.linspace(0., 1., args.ais_steps)
    
    generated_count = 0
    print(f"Start generating {total_samples} images using AIS...")
    print(f" - Steps per batch: {args.ais_steps}")
    print(f" - Total batches: {num_batches}")

    for b in range(num_batches):
        current_batch_size = min(batch_size, total_samples - generated_count)
        if current_batch_size <= 0:
            break
        
        print(f"Batch {b+1}/{num_batches} (Size: {current_batch_size})")

        x_sample = init_dist.sample((current_batch_size,))
        
        for itr, beta_k in enumerate(tqdm(betas, desc=f"AIS Annealing ({b+1})", leave=False)):
            model_k = lambda x: ais_model(x, beta=beta_k)
            
            x_sample = sampler.step(x_sample.detach(), model_k).detach()
        
        print(f"Saving batch {b+1} images...")
        for i in range(current_batch_size):
            img = x_sample[i].cpu()
            img = img.view(1, args.input_size[1], args.input_size[2])
            
            file_name = f"ais_sample_{generated_count:05d}.png"
            save_path = os.path.join(args.save_dir, file_name)
            
            torchvision.utils.save_image(img, save_path, normalize=True)
            generated_count += 1

    print(f"Done! {generated_count} AIS images saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='Folder to save images')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    
    parser.add_argument('--dataset_name', type=str, default='static_mnist', choices=['static_mnist', 'dynamic_mnist', 'omniglot', 'caltech', 'freyfaces', 'cifar10'])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    
    parser.add_argument('--n_samples', type=int, default=2000, help='Total number of images to generate')
    parser.add_argument('--ais_steps', type=int, default=10000, help='Number of annealing steps (higher is better quality)')
    
    parser.add_argument('--model', type=str, default='resnet-64')
    parser.add_argument('--sampler', type=str, default='gwg')
    parser.add_argument('--alpha', type=float, default=1.0, help='For ScoreTiltedMCMC')
    
    parser.add_argument('--base_dist', action='store_true', default=True)
    parser.add_argument('--ema', action='store_true', help='Use EMA model weights')
    parser.add_argument('--seed', type=int, default=1234)
    
    parser.add_argument('--n_out', type=int, default=3)
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--steps_per_iter', type=int, default=1)
    
    args = parser.parse_args()
    main(args)