import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision.datasets import VOCDetection
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import gc
from collections import defaultdict
from tqdm.notebook import tqdm  # Use notebook tqdm for Kaggle
from sklearn.metrics import average_precision_score
# Mixed Precision
from torch.cuda.amp import GradScaler, autocast
import random

# --- Configuration ---
CONFIG = {
    "data_dir": "./VOCdevkit/VOC2008",  # Updated to match VOC structure
    "year": "2008",
    "image_set_train": "train",
    "image_set_val": "val",
    "classes": [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ],
    "num_classes": 20,
    "gan_img_size": 64, # Smaller size for GAN training feasibility
    "classifier_img_size": 224, # Standard size for ResNet
    "gan_nz": 100,       # Size of z latent vector (i.e. size of generator input)
    "gan_ngf": 64,       # Size of feature maps in generator
    "gan_ndf": 64,       # Size of feature maps in discriminator
    "gan_epochs": 150,   # Increased from 50 to 150 for better convergence
    "gan_batch_size": 32, # Reduced from 64 for more stable gradients
    "gan_lr_g": 0.0001,  # Lower learning rate for generator (was 0.0002)
    "gan_lr_d": 0.00005, # Even lower learning rate for discriminator to prevent it from overwhelming the generator
    "gan_beta1": 0.5,    # Beta1 for Adam optimizer
    "gan_label_smoothing": 0.1, # Add label smoothing to help stabilize training
    "gan_early_stopping_patience": 20,  # Number of epochs for GAN early stopping
    "gan_early_stopping_threshold": 0.05,  # Relative improvement threshold for GAN
    "classifier_epochs": 25, # Adjust based on convergence & time
    "classifier_batch_size": 32, # Adjust based on GPU memory
    "classifier_lr": 0.001,
    "early_stopping_patience": 5,  # Number of epochs with no improvement to wait before stopping
    "early_stopping_min_delta": 0.001,  # Minimum change to qualify as improvement
    "augmentation_samples": [0, 100, 200, 500], # 0 is baseline
    "results_dir": "./results",
    "gan_models_dir": "./gan_models",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}

# Create directories
os.makedirs(CONFIG["results_dir"], exist_ok=True)
os.makedirs(CONFIG["gan_models_dir"], exist_ok=True)

# Utility function to download and extract VOC dataset
def download_voc_dataset():
    """Download and extract VOC2008 dataset if it doesn't exist."""
    import urllib.request
    import tarfile
    
    voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar"
    tar_file_path = "./VOCtrainval_14-Jul-2008.tar"
    
    # Download if the tar file doesn't exist
    if not os.path.exists(tar_file_path):
        print(f"Downloading VOC2008 dataset from {voc_url}...")
        try:
            urllib.request.urlretrieve(voc_url, tar_file_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print(f"Please download manually from {voc_url}")
            return False
    else:
        print(f"Found existing VOC dataset file: {tar_file_path}")
    
    # Extract the tar file
    try:
        print("Extracting dataset...")
        with tarfile.open(tar_file_path) as tar:
            tar.extractall(path=".")
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

# Check if VOC dataset structure exists
voc_data_dir = CONFIG["data_dir"]
voc_jpeg_dir = os.path.join(voc_data_dir, 'JPEGImages')
voc_imageset_dir = os.path.join(voc_data_dir, 'ImageSets', 'Main')

# If dataset structure doesn't exist, try to download it
if not os.path.exists(voc_data_dir):
    print(f"VOC dataset directory not found: {voc_data_dir}")
    if download_voc_dataset():
        print("VOC dataset has been downloaded and extracted.")
    else:
        print("Could not download VOC dataset automatically.")
        print("Please download and extract it manually to continue.")
        print("Expected structure: ./VOCdevkit/VOC2008/")

# Set seed for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])
    # Enable benchmark mode for potential performance improvements if input sizes don't vary
    torch.backends.cudnn.benchmark = True

print(f"Using device: {CONFIG['device']}")
print(f"Number of classes: {CONFIG['num_classes']}")
print(f"Class names: {CONFIG['classes']}")

# --- 1. Data Setup ---

# Custom Dataset to handle VOC for Multi-Label Classification
class VOCClassificationCustomDataset(Dataset):
    def __init__(self, base_dir, image_set, transform=None, return_img_path=False):
        # Check and adjust image directory name if needed
        self.image_dir = os.path.join(base_dir, 'JPEGImages')
        if not os.path.exists(self.image_dir):
            alternative_dirs = ['images', 'Images', 'JPEG', 'jpeg']
            for alt_dir in alternative_dirs:
                alt_path = os.path.join(base_dir, alt_dir)
                if os.path.exists(alt_path):
                    self.image_dir = alt_path
                    print(f"Using alternative image directory: {alt_path}")
                    break
        
        self.imageset_dir = os.path.join(base_dir, 'ImageSets', 'Main')
        if not os.path.exists(self.imageset_dir):
            alt_path = os.path.join(base_dir, 'ImageSets')
            if os.path.exists(alt_path):
                self.imageset_dir = alt_path
                print(f"Using alternative imageset directory: {alt_path}")
        
        self.transform = transform
        self.image_set = image_set
        self.return_img_path = return_img_path # Flag to return image path for visualization
        self.classes = CONFIG['classes']
        self.num_classes = CONFIG['num_classes']
        
        # Check if image_set.txt exists (train.txt or val.txt)
        split_file = os.path.join(self.imageset_dir, f"{image_set}.txt")
        
        # If it doesn't exist, try trainval.txt instead
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found.")
            trainval_file = os.path.join(self.imageset_dir, "trainval.txt")
            
            if os.path.exists(trainval_file):
                print(f"Using trainval.txt instead for {image_set}")
                split_file = trainval_file
            else:
                # If trainval.txt also doesn't exist, create image_set.txt from class-specific files
                print(f"Creating {image_set}.txt from class-specific files...")
                self._create_image_set_file(image_set)
                
                # Check if we successfully created the file
                if not os.path.exists(split_file):
                    raise FileNotFoundError(f"Could not create {split_file}. Check that class-specific files exist.")
        
        with open(split_file, 'r') as f: 
            self.image_ids = [line.strip() for line in f if line.strip()]
        
        self._load_labels()
        print(f"Found {len(self.image_ids)} IDs, loaded labels for Cls '{self.image_set}' set.")
        if not self.image_ids: 
            print(f"Warning: Cls image set '{image_set}' is empty.")
    
    def _create_image_set_file(self, image_set):
        """Create image_set.txt file from class-specific files"""
        # First, collect all image IDs from class-specific files
        all_image_ids = set()
        
        for class_name in self.classes:
            class_file = os.path.join(self.imageset_dir, f"{class_name}_{image_set}.txt")
            
            # If we can't find class_name_image_set.txt, try class_name_trainval.txt
            if not os.path.exists(class_file):
                class_file = os.path.join(self.imageset_dir, f"{class_name}_trainval.txt")
                
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            img_id = parts[0]
                            all_image_ids.add(img_id)
        
        # Write image IDs to image_set.txt
        if all_image_ids:
            output_file = os.path.join(self.imageset_dir, f"{image_set}.txt")
            with open(output_file, 'w') as f:
                for img_id in sorted(all_image_ids):
                    f.write(f"{img_id}\n")
            print(f"Created {output_file} with {len(all_image_ids)} image IDs.")

    def _load_labels(self):
        self.labels = {}
        class_label_data = {}
        
        for class_name in self.classes:
            # Try to find class file with format: class_image_set.txt
            class_file = os.path.join(self.imageset_dir, f"{class_name}_{self.image_set}.txt")
            
            # If it doesn't exist, try trainval
            if not os.path.exists(class_file):
                class_file = os.path.join(self.imageset_dir, f"{class_name}_trainval.txt")
            
            # Initialize mapping for this class
            image_to_label = {}
            class_label_data[class_name] = image_to_label
            
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2: 
                            img_id, label_val = parts[0], int(parts[1])
                            image_to_label[img_id] = label_val
                print(f"Loaded {len(image_to_label)} labels for class '{class_name}'")
            else:
                print(f"Warning: No label file found for class '{class_name}'")
                        
        for img_id in self.image_ids:
            label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
            for i, class_name in enumerate(self.classes):
                if class_label_data[class_name].get(img_id, 0) == 1: 
                    label_vector[i] = 1.0
            self.labels[img_id] = label_vector

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        label = self.labels.get(img_id, torch.zeros(self.num_classes, dtype=torch.float32))
        dummy_image = Image.new('RGB', (224, 224), color='grey')
        
        # Try different image extensions if .jpg doesn't exist
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpeg', '.JPG', '.JPEG', '.PNG']:
                alt_path = os.path.join(self.image_dir, f"{img_id}{ext}")
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, IOError) as e: 
            print(f"\nError: Cannot load image file: {img_path} - {e}")
            image = dummy_image

        if self.transform:
            try: 
                transformed_image = self.transform(image)
            except Exception as e:
                print(f"\nError transforming image {img_id}: {e}")
                transformed_image = self.transform(dummy_image) if self.transform else transforms.ToTensor()(dummy_image)
        else:
             transformed_image = transforms.ToTensor()(image) # Basic transform if none provided

        if self.return_img_path:
             return transformed_image, label, img_path # Return path if requested
        else:
             return transformed_image, label

# Define Transforms
# For GAN Training (smaller size, simple normalization)
gan_transform = transforms.Compose([
    transforms.Resize(CONFIG['gan_img_size']),
    transforms.CenterCrop(CONFIG['gan_img_size']),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# For Classifier Training/Evaluation (larger size, ImageNet normalization)
classifier_train_transform = transforms.Compose([
    transforms.Resize((CONFIG['classifier_img_size'], CONFIG['classifier_img_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Add color jittering for better augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classifier_val_transform = transforms.Compose([
    transforms.Resize((CONFIG['classifier_img_size'], CONFIG['classifier_img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
# --- IMPORTANT: Create separate dataset instances for GAN and Classifier ---
# --- because they use different transforms ---

# Original Training Data (for classifier baseline & augmentation source)
original_train_dataset_classifier = VOCClassificationCustomDataset(
    base_dir=CONFIG['data_dir'],
    image_set=CONFIG['image_set_train'],
    transform=classifier_train_transform
)

# Validation Data (for classifier evaluation)
val_dataset_classifier = VOCClassificationCustomDataset(
    base_dir=CONFIG['data_dir'],
    image_set=CONFIG['image_set_val'],
    transform=classifier_val_transform
)

val_loader = DataLoader(val_dataset_classifier, batch_size=CONFIG['classifier_batch_size'], shuffle=False, num_workers=2, pin_memory=True)

print(f"Original training samples (for classifier): {len(original_train_dataset_classifier)}")
print(f"Validation samples (for classifier): {len(val_dataset_classifier)}")

# --- 2. GAN Implementation (DCGAN) ---

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ndf, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),  # Add dropout for regularization
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),  # Add dropout for regularization
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# --- GAN Training Function ---
def train_gan_for_class(class_name, class_idx, data_dir, image_set, gan_transform, config):
    print(f"\n--- Training GAN for class: {class_name} ---")
    gan_model_path = os.path.join(config['gan_models_dir'], f"generator_{class_name}.pth")
    if os.path.exists(gan_model_path):
        print(f"GAN generator model already exists for {class_name}. Skipping training.")
        return

    # --- Create a dataset containing ONLY images with the target class ---
    # Use a dedicated dataset instance with GAN transforms
    gan_train_dataset_full = VOCClassificationCustomDataset(
        base_dir=data_dir,
        image_set=image_set,
        transform=gan_transform
    )

    # Filter indices for the current class
    indices = []
    for i, img_id in enumerate(gan_train_dataset_full.image_ids):
        label = gan_train_dataset_full.labels.get(img_id)
        if label is not None and label[class_idx] == 1:
            indices.append(i)

    if not indices:
        print(f"Warning: No training images found for class {class_name}. Cannot train GAN.")
        return

    print(f"Found {len(indices)} training images containing class {class_name}.")
    class_dataset = Subset(gan_train_dataset_full, indices)
    dataloader = DataLoader(class_dataset, batch_size=config['gan_batch_size'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # Initialize models
    netG = Generator(config['gan_nz'], config['gan_ngf']).to(config['device'])
    netD = Discriminator(config['gan_ndf']).to(config['device'])
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config['gan_lr_d'], betas=(config['gan_beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config['gan_lr_g'], betas=(config['gan_beta1'], 0.999))
    
    # Learning rate schedulers - reduce learning rate over time
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.5)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.5)

    # Fixed noise for visualization
    fixed_noise = torch.randn(config['gan_batch_size'], config['gan_nz'], 1, 1, device=config['device'])

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    # Instance noise - decrease over time
    noise_std = 0.1  # Initial standard deviation of noise

    # Training Loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    start_time = time.time()
    
    # Early stopping variables
    patience = config.get('gan_early_stopping_patience', 20)
    min_improvement = config.get('gan_early_stopping_threshold', 0.05)
    best_g_loss = float('inf')
    best_epoch = 0
    stop_training = False
    window_size = 10  # Window size for loss smoothing

    print("Starting GAN Training Loop...")
    for epoch in range(config['gan_epochs']):
        if stop_training:
            print(f"Early stopping GAN training at epoch {epoch}")
            break
            
        epoch_start_time = time.time()
        # Decrease noise standard deviation over epochs
        current_noise_std = max(0, noise_std * (1 - epoch/config['gan_epochs']))
        
        # Track losses for this epoch
        epoch_g_losses = []
        
        for i, data in enumerate(dataloader, 0):
            # --- Train Discriminator ---
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(config['device'])
            b_size = real_cpu.size(0)
            
            # Add instance noise to real images (decreasing over time)
            if current_noise_std > 0:
                real_cpu = real_cpu + current_noise_std * torch.randn_like(real_cpu)
                # Clamp values to valid range [-1, 1]
                real_cpu = torch.clamp(real_cpu, -1, 1)
            
            # Label smoothing - use soft labels for real images
            smooth_real_labels = torch.full((b_size,), real_label - config['gan_label_smoothing'], 
                                          dtype=torch.float, device=config['device'])
            
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, smooth_real_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config['gan_nz'], 1, 1, device=config['device'])
            # Generate fake image batch with G
            fake = netG(noise)
            
            # Add instance noise to fake images (decreasing over time)
            if current_noise_std > 0:
                fake = fake + current_noise_std * torch.randn_like(fake)
                # Clamp values to valid range [-1, 1]
                fake = torch.clamp(fake, -1, 1)
                
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=config['device'])
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1) # detach G's history
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # --- Train Generator ---
            netG.zero_grad()
            # We want to use soft labels here too (a bit less than 1.0) for stability
            soft_labels = torch.full((b_size,), real_label - config['gan_label_smoothing']/2, 
                                    dtype=torch.float, device=config['device'])
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, soft_labels)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Save generator loss for early stopping
            epoch_g_losses.append(errG.item())

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch+1}/{config["gan_epochs"]}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} Noise: {current_noise_std:.4f}')

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 200 == 0) or ((epoch == config['gan_epochs']-1) and (i == len(dataloader)-1)):
                 with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                 img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                 # Save grid image
                 grid_filename = os.path.join(config['results_dir'], f"gan_progress_{class_name}_epoch{epoch+1}.png")
                 vutils.save_image(img_list[-1], grid_filename)

            iters += 1
            
        # Calculate average G loss for this epoch
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else float('inf')
        
        # Early stopping check - smooth loss over window if possible
        if len(G_losses) >= window_size:
            # Use a moving average of G_losses for smoother early stopping
            recent_g_losses = G_losses[-window_size:]
            smoothed_g_loss = sum(recent_g_losses) / window_size
        else:
            smoothed_g_loss = avg_g_loss
            
        # Check for improvement
        relative_improvement = (best_g_loss - smoothed_g_loss) / best_g_loss if best_g_loss > 0 else 1.0
        
        if smoothed_g_loss < best_g_loss or relative_improvement > min_improvement:
            best_g_loss = smoothed_g_loss
            best_epoch = epoch
            # Save the current best model
            torch.save(netG.state_dict(), gan_model_path)
            print(f"New best G loss: {best_g_loss:.4f} (saved model)")
        
        # Check for early stopping
        epochs_without_improvement = epoch - best_epoch
        if epoch > 30 and epochs_without_improvement >= patience:  # Use early stopping only after 30 epochs
            print(f"Early stopping triggered. No improvement for {epochs_without_improvement} epochs.")
            stop_training = True
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f} seconds. G Loss: {avg_g_loss:.4f}, Best: {best_g_loss:.4f}, Epochs without improvement: {epochs_without_improvement}")
        
        # Step the learning rate schedulers
        schedulerD.step()
        schedulerG.step()
        current_lr_g = optimizerG.param_groups[0]['lr']
        current_lr_d = optimizerD.param_groups[0]['lr']
        print(f"Learning rates: G={current_lr_g:.6f}, D={current_lr_d:.6f}")

    total_time = time.time() - start_time
    print(f"Finished GAN training for {class_name} in {total_time:.2f} seconds.")

    # Plot losses
    plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training ({class_name})")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config['results_dir'], f"gan_loss_{class_name}.png"))
    # plt.show() # Avoid showing in Kaggle script mode
    plt.close() # Close plot to free memory

    # Clean up
    del netG, netD, optimizerG, optimizerD, dataloader, class_dataset, gan_train_dataset_full
    gc.collect()
    torch.cuda.empty_cache()


# --- Function to Generate Synthetic Data ---
def generate_synthetic_data(n_samples_per_class, config):
    """
    Generate synthetic data with class-adaptive sample counts and stricter quality filtering.
    
    Args:
        n_samples_per_class: Maximum number of samples per class
        config: Configuration dictionary
    
    Returns:
        Synthetic images and labels tensors
    """
    synthetic_images = []
    synthetic_labels = []
    generator = Generator(config['gan_nz'], config['gan_ngf']).to(config['device'])

    # Class count analysis to determine adaptive sample counts
    class_counts = get_class_distribution(original_train_dataset_classifier)
    max_class_count = max(class_counts.values())
    
    # Calculate quality thresholds - stricter for rare classes
    quality_thresholds = {}
    
    print("\n--- Analyzing Class Distribution for Adaptive Sampling ---")
    print(f"{'Class':<15} {'Original Count':<15} {'Sample Target':<15} {'Quality Threshold':<15}")
    print("-" * 60)
    
    # Calculate target samples for each class based on original distribution
    target_samples = {}
    for class_name, count in class_counts.items():
        # Define how many samples to generate based on class frequency
        # Square root scaling to balance between proportional and equal sampling
        ratio = count / max_class_count
        # Use square root to reduce the disparity but maintain some proportion
        # Apply a minimum cap of 10% and max of 80% of requested samples
        scaling_factor = min(0.8, max(0.1, np.sqrt(ratio)))
        target_count = min(count, int(n_samples_per_class * scaling_factor))
        
        # More strict quality filtering for rare classes
        quality_threshold = 0.15 + (0.25 * (1 - ratio))  # 0.15 to 0.4 based on rarity
        
        target_samples[class_name] = target_count
        quality_thresholds[class_name] = quality_threshold
        
        print(f"{class_name:<15} {count:<15} {target_count:<15} {quality_threshold:.3f}")
    
    resize_normalize_transform = transforms.Compose([
        transforms.Resize((config['classifier_img_size'], config['classifier_img_size'])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Use ImageNet norm
    ])

    print(f"\n--- Generating Class-Adaptive Synthetic Samples ---")
    generated_samples_count = 0  # Track total generated samples
    
    for class_idx, class_name in enumerate(tqdm(config['classes'])):
        # Skip if no samples requested for this class
        if target_samples[class_name] <= 0:
            print(f"Skipping {class_name} - no samples requested")
            continue
            
        gan_model_path = os.path.join(config['gan_models_dir'], f"generator_{class_name}.pth")
        if not os.path.exists(gan_model_path):
            print(f"Warning: Generator model not found for {class_name}. Skipping generation for this class.")
            continue

        generator.load_state_dict(torch.load(gan_model_path, map_location=config['device']))
        generator.eval()
        
        # Use a larger batch of noise and select best samples
        selection_factor = 3  # Generate 3x samples and select best ones
        generation_batch_size = min(64, target_samples[class_name] * selection_factor)
        
        generated_count = 0
        quality_threshold = quality_thresholds[class_name]
        
        with torch.no_grad():
            attempts = 0
            max_attempts = 5  # Limit total attempts to prevent infinite loops
            
            while generated_count < target_samples[class_name] and attempts < max_attempts:
                batch_size = min(generation_batch_size, target_samples[class_name] - generated_count)
                if batch_size <= 0: break
                attempts += 1

                # Generate more samples than needed and select the best ones
                noise = torch.randn(batch_size * selection_factor, config['gan_nz'], 1, 1, device=config['device'])
                # Generate images in [-1, 1] range (due to Tanh)
                gen_imgs_gan_norm = generator(noise)

                # --- IMPORTANT: Post-process generated images ---
                # 1. Denormalize from GAN's [-1, 1] to [0, 1]
                gen_imgs_0_1 = gen_imgs_gan_norm * 0.5 + 0.5
                
                # Calculate quality metrics
                quality_scores = []
                for img in gen_imgs_0_1:
                    # Standard deviation - higher means more variation
                    std_score = torch.std(img).item()  
                    
                    # Calculate perceptual features like contrast, edges
                    # Convert to numpy and greyscale for edge detection
                    img_np = img.mean(dim=0).cpu().numpy()  # Average RGB channels
                    
                    # Check sharpness using Laplacian variance
                    if img_np.std() > 0.01:  # Avoid division by zero or very flat images
                        # Approximate a sharpness score based on local variations
                        laplacian = np.abs(img_np[1:, 1:] - img_np[:-1, :-1]).mean()
                        sharpness_score = min(1.0, laplacian * 10)  # Scale and cap
                    else:
                        sharpness_score = 0
                    
                    # Combined quality score (customize weights as needed)
                    combined_score = (std_score * 0.7) + (sharpness_score * 0.3)
                    quality_scores.append(combined_score)
                
                # Select top quality images that exceed the threshold
                good_indices = [i for i, score in enumerate(quality_scores) if score > quality_threshold]
                
                # If we don't have enough good samples, take the best ones available
                if len(good_indices) < batch_size:
                    # Sort by quality and take the top ones
                    indices = sorted(range(len(quality_scores)), key=lambda i: quality_scores[i], reverse=True)
                    selected_indices = indices[:batch_size]
                else:
                    # Randomly select from the good indices
                    np.random.shuffle(good_indices)
                    selected_indices = good_indices[:batch_size]
                
                selected_imgs = gen_imgs_0_1[selected_indices]
                
                # Resize to classifier input size and apply classifier normalization
                processed_imgs = torch.stack([resize_normalize_transform(img) for img in selected_imgs.cpu()]).to(config['device'])

                # Create labels (one-hot for the generated class)
                label_vector = torch.zeros(config['num_classes'], device=config['device'])
                label_vector[class_idx] = 1.0
                batch_labels = label_vector.repeat(len(selected_indices), 1)

                synthetic_images.append(processed_imgs.cpu())
                synthetic_labels.append(batch_labels.cpu())
                
                generated_count += len(selected_indices)
                generated_samples_count += len(selected_indices)
                
            print(f"Generated {generated_count}/{target_samples[class_name]} samples for {class_name}")

    # Clean up generator from GPU memory
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    if not synthetic_images:
        return None, None

    all_synthetic_images = torch.cat(synthetic_images, dim=0)
    all_synthetic_labels = torch.cat(synthetic_labels, dim=0)
    print(f"Generated {len(all_synthetic_images)} total synthetic samples across {len(config['classes'])} classes.")
    return all_synthetic_images, all_synthetic_labels

def get_class_distribution(dataset):
    """
    Count the instances of each class in the dataset.
    
    Args:
        dataset: The dataset to analyze
        
    Returns:
        A dictionary mapping class names to counts
    """
    class_counts = {class_name: 0 for class_name in CONFIG['classes']}
    
    # Iterate through dataset and count class occurrences
    for i in range(len(dataset)):
        _, label = dataset[i]
        for class_idx, class_name in enumerate(CONFIG['classes']):
            if label[class_idx] == 1:
                class_counts[class_name] += 1
    
    return class_counts

# --- Custom Dataset for Augmented Data ---
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, synthetic_images, synthetic_labels, synthetic_weight=0.3):
        """
        Create an augmented dataset that combines original and synthetic data
        with class-specific weighting.
        
        Args:
            original_dataset: Dataset containing original training data
            synthetic_images: Tensor of synthetic images generated by GAN
            synthetic_labels: Tensor of labels for synthetic images
            synthetic_weight: Base weight to give synthetic samples (0.0-1.0)
        """
        self.original_dataset = original_dataset
        self.synthetic_images = synthetic_images
        self.synthetic_labels = synthetic_labels
        
        # Limit the maximum synthetic weight to prevent overwhelming the model
        self.synthetic_weight = min(0.5, max(0.05, synthetic_weight))
        
        if synthetic_images is None or synthetic_labels is None:
            self.synthetic_len = 0
        else:
            self.synthetic_len = len(synthetic_images)

        self.original_len = len(original_dataset)
        self.total_len = self.original_len + self.synthetic_len
        
        # Calculate class-specific weights
        self.calculate_class_weights()
        
        # Create sample weights for balanced sampling
        self.create_sample_weights()
        
        print(f"Created augmented dataset with {self.original_len} original and {self.synthetic_len} synthetic samples")
        print(f"Using base synthetic weight of {self.synthetic_weight:.2f}")
    
    def calculate_class_weights(self):
        """Calculate class-specific weights based on frequency in original dataset"""
        self.class_counts_original = {i: 0 for i in range(CONFIG['num_classes'])}
        self.class_counts_synthetic = {i: 0 for i in range(CONFIG['num_classes'])}
        
        # Count original samples per class
        for i in range(self.original_len):
            _, label = self.original_dataset[i]
            for class_idx in range(CONFIG['num_classes']):
                if label[class_idx] == 1:
                    self.class_counts_original[class_idx] += 1
        
        # Count synthetic samples per class
        if self.synthetic_len > 0:
            for i in range(self.synthetic_len):
                label = self.synthetic_labels[i]
                for class_idx in range(CONFIG['num_classes']):
                    if label[class_idx] == 1:
                        self.class_counts_synthetic[class_idx] += 1
        
        # Calculate weight adjustments for each class
        self.class_weights = {}
        for class_idx in range(CONFIG['num_classes']):
            orig_count = max(1, self.class_counts_original[class_idx])
            synth_count = self.class_counts_synthetic[class_idx]
            
            # For rare classes (low count), reduce synthetic weight
            if orig_count < 50:
                # Very rare class - use very little synthetic data
                weight_multiplier = 0.3
            elif orig_count < 100:
                # Somewhat rare - use moderate synthetic data
                weight_multiplier = 0.5
            else:
                # Common class - can use more synthetic data
                weight_multiplier = 0.8
                
            # Adjust synthetic weight for this class
            self.class_weights[class_idx] = max(0.05, min(0.7, self.synthetic_weight * weight_multiplier))
    
    def create_sample_weights(self):
        """Create weights for each sample to support balanced sampling"""
        if self.synthetic_len == 0:
            return
            
        # Initialize arrays to store sample indices and their weights
        self.synthetic_class_indices = {i: [] for i in range(CONFIG['num_classes'])}
        
        # Group synthetic samples by class
        for i in range(self.synthetic_len):
            label = self.synthetic_labels[i]
            for class_idx in range(CONFIG['num_classes']):
                if label[class_idx] == 1:
                    self.synthetic_class_indices[class_idx].append(i)
    
    def analyze_class_distribution(self):
        """Analyze and report class distribution in both datasets"""
        if self.synthetic_len == 0:
            return
            
        print("\nClass distribution analysis:")
        print(f"{'Class':<15} {'Original':<10} {'Synthetic':<10} {'Ratio':<10} {'Weight':<10}")
        print("-" * 55)
        
        for class_idx, class_name in enumerate(CONFIG['classes']):
            orig = self.class_counts_original[class_idx]
            synth = self.class_counts_synthetic[class_idx]
            ratio = synth / max(1, orig)  # Avoid division by zero
            weight = self.class_weights[class_idx]
            print(f"{class_name:<15} {int(orig):<10} {int(synth):<10} {ratio:.2f}{' ':<10}{weight:.2f}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.original_len:
            # Get from original dataset with probability 1 - synthetic_weight
            # For class-specific weighting, we'll also consider the classes in this sample
            sample_img, sample_label = self.original_dataset[idx]
            
            # Find which classes are present in this sample
            present_classes = [i for i in range(CONFIG['num_classes']) if sample_label[i] == 1]
            
            if present_classes and self.synthetic_len > 0:
                # Calculate average weight for classes in this sample
                avg_weight = sum(self.class_weights[c] for c in present_classes) / len(present_classes)
                
                # Use this weight to decide whether to use a synthetic sample instead
                if random.random() < avg_weight:
                    # Choose a class from present classes to sample from
                    target_class = random.choice(present_classes)
                    
                    # If we have synthetic samples for this class
                    if self.synthetic_class_indices[target_class]:
                        # Sample from the synthetic data for this class
                        synth_idx = random.choice(self.synthetic_class_indices[target_class])
                        return self.synthetic_images[synth_idx], self.synthetic_labels[synth_idx]
            
            # Return the original sample
            return sample_img, sample_label
        else:
            # Default case - get from synthetic data with random sampling
            synthetic_idx = random.randint(0, self.synthetic_len - 1)
            return self.synthetic_images[synthetic_idx], self.synthetic_labels[synthetic_idx]


# --- 3. Classifier Implementation ---

def get_classifier(num_classes, pretrained=True):
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Replace the final layer
    return model

# --- Classifier Training Function ---
def train_classifier(model, train_loader, val_loader, config):
    model.to(config['device'])
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['classifier_lr'])
    # Learning rate scheduler (optional but often helpful)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # GradScaler for Mixed Precision
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_map = 0.0
    best_model_state = None
    
    # Early stopping variables
    patience = config.get('early_stopping_patience', 5)
    min_delta = config.get('early_stopping_min_delta', 0.001)
    counter = 0
    early_stop = False

    print("\n--- Starting Classifier Training ---")
    for epoch in range(config['classifier_epochs']):
        if early_stop:
            print(f"Early stopping triggered after {epoch} epochs!")
            break
            
        model.train()
        running_loss = 0.0
        train_start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['classifier_epochs']} [Train]")
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])

            optimizer.zero_grad()

            # Mixed Precision Context
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
            # Otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")


        epoch_loss = running_loss / len(train_loader.dataset)
        train_time = time.time() - train_start_time

        # Validation phase (using evaluate_classifier for mAP)
        val_map, val_loss = evaluate_classifier(model, val_loader, criterion, config)

        print(f"Epoch {epoch+1}/{config['classifier_epochs']} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val mAP: {val_map:.4f}, "
              f"Train Time: {train_time:.2f}s")

        scheduler.step() # Step the scheduler

        # Save the best model based on validation mAP
        if val_map > best_map + min_delta:
            best_map = val_map
            # Save model state on CPU to avoid GPU memory issues when loading later
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"*** New best model saved with mAP: {best_map:.4f} ***")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                early_stop = True
                print("Early stopping: Validation mAP did not improve for", patience, "epochs.")

    print("Finished Classifier Training.")
    # Load the best model state back
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state with mAP: {best_map:.4f}")
    else:
         print("Warning: No best model state was saved (validation mAP might not have improved).")

    return model, best_map # Return trained model and best mAP

# --- Evaluation Function (Calculate mAP) ---
def calculate_ap(y_true, y_scores):
    """Calculate Average Precision (AP) for a single class."""
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Handle cases where a class is not present in the ground truth
    if np.sum(y_true) == 0:
        # print("Warning: No positive samples found for this class in validation set. Returning AP=0.")
        return 0.0 # Or np.nan, depending on how you want to handle this

    return average_precision_score(y_true, y_scores)

def calculate_map(all_labels, all_preds):
    """Calculate Mean Average Precision (mAP) across all classes."""
    num_classes = all_labels.shape[1]
    average_precisions = []
    for i in range(num_classes):
        ap = calculate_ap(all_labels[:, i], all_preds[:, i])
        average_precisions.append(ap)

    # Filter out potential NaN values if any class had no positive samples
    valid_aps = [ap for ap in average_precisions if not np.isnan(ap)]
    if not valid_aps:
        return 0.0 # Or np.nan
    return np.mean(valid_aps)


def evaluate_classifier(model, dataloader, criterion, config):
    model.to(config['device'])
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0

    print("--- Evaluating Classifier ---")
    progress_bar = tqdm(dataloader, desc="Evaluation")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])

            # Mixed precision inference
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Calculate loss if needed

            running_loss += loss.item() * inputs.size(0)

            # Store predictions (probabilities using sigmoid) and true labels
            # Move to CPU before converting to numpy to avoid GPU sync issues
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels_np)

    # Concatenate all batch results
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate mAP
    mean_ap = calculate_map(all_labels, all_preds)
    epoch_loss = running_loss / len(dataloader.dataset)

    return mean_ap, epoch_loss

# --- Save Sample Images from Each Class's GAN ---

def save_gan_samples(config, num_samples=5):
    """
    Save sample images generated by each class's GAN generator.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of samples to save per class
    """
    print("\n--- Saving GAN Sample Images ---")
    
    # Create a directory for GAN samples
    samples_dir = os.path.join(config['results_dir'], "gan_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Initialize generator
    generator = Generator(config['gan_nz'], config['gan_ngf']).to(config['device'])
    
    # Define inverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )
    
    for class_idx, class_name in enumerate(config['classes']):
        gan_model_path = os.path.join(config['gan_models_dir'], f"generator_{class_name}.pth")
        if not os.path.exists(gan_model_path):
            print(f"Warning: Generator model not found for {class_name}. Skipping sample generation.")
            continue
            
        print(f"Generating samples for class: {class_name}")
        
        # Load the trained generator
        generator.load_state_dict(torch.load(gan_model_path, map_location=config['device']))
        generator.eval()
        
        # Generate samples
        with torch.no_grad():
            # Generate a batch of images
            noise = torch.randn(num_samples, config['gan_nz'], 1, 1, device=config['device'])
            fake_images = generator(noise)
            
            # Denormalize for visualization
            fake_images = inv_normalize(fake_images)
            
            # Save individual images
            for i in range(num_samples):
                img_path = os.path.join(samples_dir, f"{class_name}_sample_{i+1}.png")
                vutils.save_image(fake_images[i], img_path)
            
            # Save a grid of all samples for this class
            grid_path = os.path.join(samples_dir, f"{class_name}_grid.png")
            grid = vutils.make_grid(fake_images, nrow=num_samples, padding=2, normalize=True)
            vutils.save_image(grid, grid_path)
    
    print(f"GAN samples saved to {samples_dir}")
    del generator
    gc.collect()
    torch.cuda.empty_cache()

def save_synthetic_samples(synthetic_images, synthetic_labels, config, num_samples=5):
    """
    Save sample images from the synthetic data generation process.
    
    Args:
        synthetic_images: Tensor of synthetic images
        synthetic_labels: Tensor of synthetic labels
        config: Configuration dictionary
        num_samples: Number of samples to save per class
    """
    print("\n--- Saving Synthetic Sample Images ---")
    
    # Create a directory for synthetic samples
    samples_dir = os.path.join(config['results_dir'], "synthetic_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Define inverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Convert labels to numpy for easier handling
    labels_np = synthetic_labels.numpy()
    
    # For each class, save a few samples
    for class_idx, class_name in enumerate(config['classes']):
        # Find indices of images with this class
        class_indices = np.where(labels_np[:, class_idx] == 1)[0]
        
        if len(class_indices) == 0:
            print(f"No synthetic samples found for class {class_name}")
            continue
            
        # Take a random subset if we have more than num_samples
        if len(class_indices) > num_samples:
            selected_indices = np.random.choice(class_indices, num_samples, replace=False)
        else:
            selected_indices = class_indices
            
        # Get the selected images
        selected_images = synthetic_images[selected_indices]
        
        # Denormalize for visualization
        denormalized_images = inv_normalize(selected_images)
        
        # Save individual images
        for i, img_idx in enumerate(selected_indices):
            img_path = os.path.join(samples_dir, f"{class_name}_synthetic_{i+1}.png")
            vutils.save_image(denormalized_images[i], img_path)
        
        # Save a grid of all samples for this class
        grid_path = os.path.join(samples_dir, f"{class_name}_synthetic_grid.png")
        grid = vutils.make_grid(denormalized_images, nrow=min(num_samples, len(selected_indices)), padding=2, normalize=True)
        vutils.save_image(grid, grid_path)
    
    print(f"Synthetic samples saved to {samples_dir}")

def save_original_samples(dataset, config, num_samples=5):
    """
    Save sample images from the original training dataset.
    
    Args:
        dataset: The original training dataset
        config: Configuration dictionary
        num_samples: Number of samples to save per class
    """
    print("\n--- Saving Original Training Sample Images ---")
    
    # Create a directory for original samples
    samples_dir = os.path.join(config['results_dir'], "original_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Define inverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Create a dictionary to store indices for each class
    class_indices = {i: [] for i in range(config['num_classes'])}
    
    # Collect indices for each class
    for i in range(len(dataset)):
        _, label = dataset[i]
        for class_idx in range(config['num_classes']):
            if label[class_idx] == 1:
                class_indices[class_idx].append(i)
    
    # For each class, save a few samples
    for class_idx, class_name in enumerate(config['classes']):
        indices = class_indices[class_idx]
        
        if len(indices) == 0:
            print(f"No original samples found for class {class_name}")
            continue
            
        # Take a random subset if we have more than num_samples
        if len(indices) > num_samples:
            selected_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            selected_indices = indices
            
        # Get the selected images
        selected_images = []
        for idx in selected_indices:
            img, _ = dataset[idx]
            selected_images.append(img)
        
        selected_images = torch.stack(selected_images)
        
        # Denormalize for visualization
        denormalized_images = inv_normalize(selected_images)
        
        # Save individual images
        for i, img_idx in enumerate(selected_indices):
            img_path = os.path.join(samples_dir, f"{class_name}_original_{i+1}.png")
            vutils.save_image(denormalized_images[i], img_path)
        
        # Save a grid of all samples for this class
        grid_path = os.path.join(samples_dir, f"{class_name}_original_grid.png")
        grid = vutils.make_grid(denormalized_images, nrow=min(num_samples, len(selected_indices)), padding=2, normalize=True)
        vutils.save_image(grid, grid_path)
    
    print(f"Original samples saved to {samples_dir}")

# --- Main Experiment Loop ---

# 1. (Optional but Recommended) Train GANs first and save models
# Set this to False if GAN models already exist
TRAIN_GANS = True
if TRAIN_GANS:
    for idx, name in enumerate(CONFIG['classes']):
        train_gan_for_class(
            class_name=name,
            class_idx=idx,
            data_dir=CONFIG['data_dir'],
            image_set=CONFIG['image_set_train'],
            gan_transform=gan_transform,
            config=CONFIG
        )
    
    # Save sample images from each class's GAN
    save_gan_samples(CONFIG, num_samples=5)
else:
    print("Skipping GAN training as TRAIN_GANS is set to False.")

# 2. Train and Evaluate Classifier with different augmentation levels
results_map = {}

for n_samples in CONFIG['augmentation_samples']:
    print(f"\n===== Experiment: {n_samples} Synthetic Samples Per Class =====")

    # --- Prepare Augmented Dataset ---
    if n_samples > 0:
        synthetic_images, synthetic_labels = generate_synthetic_data(n_samples, CONFIG)
        if synthetic_images is None or synthetic_labels is None:
             print(f"Skipping {n_samples} samples experiment due to generation issues.")
             # results_map[n_samples] = float('nan') # Record failure
             gc.collect()
             torch.cuda.empty_cache()
             continue # Skip to next sample count
        
        # Save synthetic samples for visualization
        save_synthetic_samples(synthetic_images, synthetic_labels, CONFIG, num_samples=5)

        # Use TensorDataset for synthetic data for easy combination
        synthetic_dataset = torch.utils.data.TensorDataset(synthetic_images, synthetic_labels)
        
        # Combine original and synthetic datasets with proper weighting
        # Start with very low synthetic weight and gradually increase with more samples
        # but keep it much lower than before
        synthetic_weight = min(0.3, 0.1 + (n_samples / 2000))  # Cap at 0.3 max weight
        augmented_dataset = AugmentedDataset(
            original_dataset=original_train_dataset_classifier, 
            synthetic_images=synthetic_images, 
            synthetic_labels=synthetic_labels,
            synthetic_weight=synthetic_weight
        )
        
        # Analyze and print class distribution
        augmented_dataset.analyze_class_distribution()
        
        train_loader = DataLoader(
            augmented_dataset, 
            batch_size=CONFIG['classifier_batch_size'], 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        print(f"Augmented training set size: {len(augmented_dataset)}")
        # Clean up large tensors
        del synthetic_images, synthetic_labels, synthetic_dataset
    else:
        # Baseline (no augmentation)
        print(f"Using original training set size: {len(original_train_dataset_classifier)}")
        train_loader = DataLoader(original_train_dataset_classifier, batch_size=CONFIG['classifier_batch_size'], shuffle=True, num_workers=2, pin_memory=True)

    # --- Train Classifier ---
    # Initialize a new classifier for each experiment to ensure fair comparison
    classifier = get_classifier(CONFIG['num_classes'], pretrained=True)
    trained_classifier, best_val_map_during_train = train_classifier(classifier, train_loader, val_loader, CONFIG)

    # --- Evaluate Final Model on Validation Set ---
    # We could re-evaluate the best loaded model, or just use the best mAP recorded during training
    # final_map, _ = evaluate_classifier(trained_classifier, val_loader, nn.BCEWithLogitsLoss(), CONFIG)
    final_map = best_val_map_during_train # Use the best mAP achieved during training epochs
    print(f"--- Final mAP for {n_samples} samples/class: {final_map:.4f} ---")
    results_map[n_samples] = final_map

    # --- Clean up GPU memory before next iteration ---
    del classifier, trained_classifier, train_loader
    if 'augmented_dataset' in locals(): del augmented_dataset # Delete if it exists
    gc.collect()
    torch.cuda.empty_cache()
    print("-" * 50)

# --- 4. Plotting and Discussion ---

print("\n===== Final Results =====")
sample_counts = sorted(results_map.keys())
map_scores = [results_map[s] for s in sample_counts]

for s, m in zip(sample_counts, map_scores):
    print(f"Samples per class: {s}, mAP: {m:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sample_counts, map_scores, marker='o')
plt.title('Classifier mAP vs. Number of Synthetic Samples per Class')
plt.xlabel('Number of Synthetic Samples per Class Added')
plt.ylabel('Mean Average Precision (mAP) on Validation Set')
plt.xticks(sample_counts) # Ensure x-axis ticks match the tested values
plt.grid(True)
plt.ylim(bottom=max(0, min(map_scores) - 0.05), top=max(map_scores) + 0.05) # Adjust y-axis limits
results_plot_path = os.path.join(CONFIG['results_dir'], "map_vs_augmentation.png")
plt.savefig(results_plot_path)
print(f"Results plot saved to {results_plot_path}")
# plt.show() # Avoid showing in Kaggle script mode
plt.close()

# Discussion
print("\n===== Discussion =====")
baseline_map = results_map[0]
print(f"Baseline mAP (0 synthetic samples): {baseline_map:.4f}")

improvement = False
best_map = baseline_map
best_samples = 0
for s in sample_counts:
    if s > 0 and results_map[s] > best_map:
        improvement = True
        best_map = results_map[s]
        best_samples = s

if improvement:
    print(f"Performance IMPROVED with augmentation.")
    print(f"Best mAP: {best_map:.4f} achieved with {best_samples} synthetic samples per class.")
    print("\nPossible reasons for improvement:")
    print("1. Increased Data Diversity: Even low-quality GAN samples might introduce variations not present in the original limited training data, helping the classifier generalize better.")
    print("2. Regularization Effect: Adding noisy or slightly different synthetic data can act as a form of regularization, preventing the classifier from overfitting to the original training samples.")
    print("3. Balancing Classes (Implicitly): While we added the same number per class, if some classes had very few original samples, the relative increase from synthetic data might be larger, potentially helping the classifier learn those classes better (although mAP averages across all).")
else:
    print(f"Performance DID NOT IMPROVE (or worsened) with augmentation.")
    print(f"Best mAP among augmented runs was {best_map:.4f} with {best_samples} samples (compare to baseline {baseline_map:.4f}).")
    print("\nPossible reasons for lack of improvement or worsening:")
    print("1. Poor GAN Quality: The DCGAN might have generated unrealistic or low-quality images (mode collapse, artifacts) that confused the classifier or didn't represent the true data distribution.")
    print("2. Distribution Shift: The synthetic data distribution might be significantly different from the real data distribution, leading the classifier astray.")
    print("3. Simplistic Labeling: Assigning only a single label to synthetic images might contradict the multi-label nature of the real data, potentially harming performance on complex images.")
    print("4. Sufficient Original Data: The original training set might already be large or diverse enough for the chosen classifier architecture, making augmentation less impactful or even detrimental if the synthetic data is noisy.")
    print("5. Suboptimal Augmentation Amount: The tested amounts (100, 200, 500) might not be optimal. Too few samples might not help, while too many poor-quality samples could hurt.")

print("\n--- Experiment Complete ---")
