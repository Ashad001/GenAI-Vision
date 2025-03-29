  
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from torchvision.datasets import VOCDetection
from PIL import Image
from sklearn.metrics import average_precision_score
import time
from tqdm import tqdm
import random
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and hyperparameters
DATA_DIR = './data/VOCdevkit/VOC2008'  # Update this to your data directory
BATCH_SIZE = 64
GAN_BATCH_SIZE = 32
NUM_EPOCHS = 30
GAN_NUM_EPOCHS = 100
LR = 0.0001
GAN_LR = 0.0002
Z_DIM = 100
IMG_SIZE = 224  # VGGNet input size

# Pascal VOC Classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(VOC_CLASSES)

# Data Preprocessing and Loading Functions
class PascalVOCClassification(Dataset):
    def __init__(self, root, year='2008', image_set='train', transform=None):
        self.voc_dataset = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=False,  # Assuming data is already downloaded
            transform=None
        )
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Process dataset to extract classification labels
        for idx in range(len(self.voc_dataset)):
            img, annotation = self.voc_dataset[idx]
            objects = annotation['annotation']['object']
            
            # Handle case where there's only one object (not in a list)
            if not isinstance(objects, list):
                objects = [objects]
            
            # Extract class labels
            labels = torch.zeros(NUM_CLASSES)
            for obj in objects:
                class_name = obj['name']
                if class_name in VOC_CLASSES:
                    class_idx = VOC_CLASSES.index(class_name)
                    labels[class_idx] = 1
            
            # Only include samples with at least one valid class
            if labels.sum() > 0:
                self.samples.append(img)
                self.targets.append(labels)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

# Define transforms for training and validation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
def load_datasets():
    print("Loading Pascal VOC datasets...")
    train_dataset = PascalVOCClassification(
        root='./data',
        image_set='train',
        transform=train_transform
    )
    
    val_dataset = PascalVOCClassification(
        root='./data',
        image_set='val',
        transform=val_transform
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

# Create data loaders
def create_data_loaders(train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

# GAN Implementation
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, channels=3, features_g=64, num_classes=NUM_CLASSES):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Embedding for class conditioning
        self.label_embed = nn.Embedding(num_classes, z_dim)
        
        self.gen = nn.Sequential(
            # Input: latent vector z concatenated with class embedding
            self._block(z_dim * 2, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            self._block(features_g * 2, features_g, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(
                features_g, channels, kernel_size=4, stride=2, padding=1  # 128x128
            ),
            nn.Tanh(),  # Output in range [-1, 1]
        )
        
        # Additional upsampling to reach 224x224
        self.upsample = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=True)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, z, labels):
        # Get one-hot index of the class
        label_idx = torch.argmax(labels, dim=1)
        
        # Get class embedding
        c_emb = self.label_embed(label_idx)
        
        # Concatenate noise and class embedding
        z_c = torch.cat([z, c_emb], dim=1)
        
        # Reshape for conv layers
        z_c = z_c.view(z_c.shape[0], z_c.shape[1], 1, 1)
        
        # Generate image
        img = self.gen(z_c)
        
        # Upsample to target size
        img = self.upsample(img)
        
        return img

class Discriminator(nn.Module):
    def __init__(self, channels=3, features_d=64, num_classes=NUM_CLASSES):
        super(Discriminator, self).__init__()
        
        # Downsampling to reduce from 224x224 to 128x128
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Main discriminator network
        self.disc = nn.Sequential(
            self._block(features_d, features_d * 2, 4, 2, 1),  # 64x64
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 32x32
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 16x16
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 8x8
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),  # 1x1
        )
        
        # Projection for class conditioning
        self.embed = nn.Embedding(num_classes, features_d * 16 * 4 * 4)
        
        # Final fully connected layer
        self.fc = nn.Linear(features_d * 16 + 1, 1)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, img, labels):
        # Downsample image
        img = self.downsample(img)
        
        # Get features from discriminator
        features = self.disc(img)
        
        # Get one-hot index of the class
        label_idx = torch.argmax(labels, dim=1)
        
        # Get class embedding and reshape
        c_emb = self.embed(label_idx).view(img.shape[0], -1)
        
        # Calculate dot product of features and class embedding
        features = features.view(img.shape[0], -1)
        
        # Concatenate features and dot product
        out = torch.cat([features, c_emb.sum(1, keepdim=True)], dim=1)
        
        # Final classification
        out = self.fc(out)
        
        return out

# Class-specific GAN trainer
class ClassConditionalGAN:
    def __init__(self, class_idx, train_dataset):
        self.class_idx = class_idx
        self.class_name = VOC_CLASSES[class_idx]
        
        # Filter dataset for samples containing this class
        self.class_indices = []
        for idx, (_, target) in enumerate(train_dataset):
            if target[class_idx] == 1:
                self.class_indices.append(idx)
        
        print(f"Class {self.class_name} has {len(self.class_indices)} samples")
        
        # Create subset dataset
        self.class_dataset = Subset(train_dataset, self.class_indices)
        self.dataloader = DataLoader(
            self.class_dataset, 
            batch_size=GAN_BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize models
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=GAN_LR, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=GAN_LR, betas=(0.5, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train(self, num_epochs=GAN_NUM_EPOCHS):
        # Training loop
        print(f"Training GAN for class {self.class_name}")
        
        for epoch in range(num_epochs):
            total_d_loss = 0
            total_g_loss = 0
            num_batches = 0
            
            for imgs, labels in self.dataloader:
                imgs = imgs.to(device)
                
                # Create one-hot vector for this class
                batch_size = imgs.shape[0]
                class_labels = torch.zeros(batch_size, NUM_CLASSES).to(device)
                class_labels[:, self.class_idx] = 1
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real images
                d_real = self.discriminator(imgs, class_labels)
                d_real_loss = self.criterion(
                    d_real, torch.ones_like(d_real)
                )
                
                # Fake images
                z = torch.randn(batch_size, Z_DIM).to(device)
                fake_imgs = self.generator(z, class_labels)
                d_fake = self.discriminator(fake_imgs.detach(), class_labels)
                d_fake_loss = self.criterion(
                    d_fake, torch.zeros_like(d_fake)
                )
                
                # Combined loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                # Generate new fake images
                z = torch.randn(batch_size, Z_DIM).to(device)
                fake_imgs = self.generator(z, class_labels)
                d_fake = self.discriminator(fake_imgs, class_labels)
                
                # Generator wants discriminator to think these are real
                g_loss = self.criterion(d_fake, torch.ones_like(d_fake))
                g_loss.backward()
                self.g_optimizer.step()
                
                # Update stats
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                num_batches += 1
            
            # Print epoch stats
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                      f"D Loss: {total_d_loss/num_batches:.4f} | "
                      f"G Loss: {total_g_loss/num_batches:.4f}")
                
                # Save sample generated images
                self.save_sample_images(epoch + 1)
    
    def save_sample_images(self, epoch, num_samples=5):
        """Save sample generated images for visualization"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, Z_DIM).to(device)
            
            # Create labels for this class
            class_labels = torch.zeros(num_samples, NUM_CLASSES).to(device)
            class_labels[:, self.class_idx] = 1
            
            # Generate images
            fake_imgs = self.generator(z, class_labels)
            
            # Denormalize images
            fake_imgs = fake_imgs * 0.5 + 0.5  # From [-1, 1] to [0, 1]
            
            # Create directory if it doesn't exist
            os.makedirs(f"gan_samples/{self.class_name}", exist_ok=True)
            
            # Save individual images
            for i in range(num_samples):
                img = fake_imgs[i].cpu().permute(1, 2, 0).numpy()
                plt.figure(figsize=(3, 3))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"gan_samples/{self.class_name}/epoch_{epoch}_sample_{i}.png")
                plt.close()
        
        self.generator.train()
    
    def generate_samples(self, num_samples=100):
        """Generate synthetic samples for this class"""
        self.generator.eval()
        gen_images = []
        gen_labels = []
        
        # Generate in batches to avoid memory issues
        batch_size = 32
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)
        
        with torch.no_grad():
            for _ in range(num_batches):
                curr_batch_size = min(batch_size, num_samples - len(gen_images))
                if curr_batch_size <= 0:
                    break
                    
                # Generate random noise
                z = torch.randn(curr_batch_size, Z_DIM).to(device)
                
                # Create labels for this class
                class_labels = torch.zeros(curr_batch_size, NUM_CLASSES).to(device)
                class_labels[:, self.class_idx] = 1
                
                # Generate images
                fake_imgs = self.generator(z, class_labels)
                
                # Store generated samples
                gen_images.append(fake_imgs.cpu())
                gen_labels.append(class_labels.cpu())
        
        # Concatenate batches
        gen_images = torch.cat(gen_images, dim=0)
        gen_labels = torch.cat(gen_labels, dim=0)
        
        self.generator.train()
        return gen_images[:num_samples], gen_labels[:num_samples]

# VGGNet Transfer Learning Model
class VGGClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGGClassifier, self).__init__()
        # Load pre-trained VGG16 model
        self.vgg = models.vgg16(pretrained=True)
        
        # Freeze convolutional layers
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return torch.sigmoid(self.vgg(x))  # Sigmoid for multi-label classification

# Training and evaluation functions
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Collect predictions and targets for mAP calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            running_loss += loss.item() * inputs.size(0)
    
    # Calculate average loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate mAP
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    ap_scores = []
    for i in range(NUM_CLASSES):
        ap = average_precision_score(all_targets[:, i], all_preds[:, i])
        ap_scores.append(ap)
    
    mean_ap = np.mean(ap_scores)
    
    return epoch_loss, mean_ap, ap_scores

def get_top_images(model, dataloader, class_idx, top_n=10):
    """Get top N images for a specific class based on prediction confidence"""
    model.eval()
    all_scores = []
    all_images = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get scores for the specified class
            scores = outputs[:, class_idx].cpu().numpy()
            
            # Store scores and images
            all_scores.extend(scores)
            all_images.extend([img.cpu() for img in inputs])
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    
    # Get indices of top N scores
    top_indices = np.argsort(all_scores)[-top_n:][::-1]
    
    # Get top images and scores
    top_images = [all_images[i] for i in top_indices]
    top_scores = all_scores[top_indices]
    
    return top_images, top_scores

# GAN augmentation function
def augment_with_gan(train_dataset, num_samples_per_class=100):
    """Generate synthetic samples using GAN for each class"""
    synthetic_images = []
    synthetic_labels = []
    
    # Create GANs for each class
    for class_idx in range(NUM_CLASSES):
        print(f"\nTraining GAN for class: {VOC_CLASSES[class_idx]}")
        
        # Create and train class-specific GAN
        class_gan = ClassConditionalGAN(class_idx, train_dataset)
        class_gan.train()
        
        # Generate synthetic samples
        gen_images, gen_labels = class_gan.generate_samples(num_samples=num_samples_per_class)
        
        # Add to collection
        synthetic_images.append(gen_images)
        synthetic_labels.append(gen_labels)
    
    # Combine all synthetic samples
    synthetic_images = torch.cat(synthetic_images, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)
    
    return synthetic_images, synthetic_labels

# Create augmented dataset class
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, synthetic_images, synthetic_labels):
        self.original_dataset = original_dataset
        self.synthetic_images = synthetic_images
        self.synthetic_labels = synthetic_labels
    
    def __len__(self):
        return len(self.original_dataset) + len(self.synthetic_images)
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            synth_idx = idx - len(self.original_dataset)
            return self.synthetic_images[synth_idx], self.synthetic_labels[synth_idx]

# Main experiment function
def run_experiment():
    # Load datasets
    train_dataset, val_dataset = load_datasets()
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)
    
    # Train baseline VGGNet
    print("\n--- Training Baseline VGGNet ---")
    baseline_model = VGGClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(baseline_model.parameters(), lr=LR)
    
    # Training loop
    best_map = 0.0
    baseline_train_losses = []
    baseline_val_losses = []
    baseline_maps = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train for one epoch
        train_loss = train_epoch(baseline_model, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        val_loss, mean_ap, ap_scores = evaluate(baseline_model, val_loader, criterion)
        
        # Record statistics
        baseline_train_losses.append(train_loss)
        baseline_val_losses.append(val_loss)
        baseline_maps.append(mean_ap)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean AP: {mean_ap:.4f}")
        
        # Save best model
        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(baseline_model.state_dict(), 'best_baseline_model.pth')
    
    # Get top images for each class from baseline model
    print("\n--- Getting Top 10 Images per Class (Baseline) ---")
    baseline_top_images = {}
    for class_idx, class_name in enumerate(VOC_CLASSES):
        top_images, top_scores = get_top_images(baseline_model, val_loader, class_idx, top_n=10)
        baseline_top_images[class_name] = (top_images, top_scores)
    
    # Save top images
    os.makedirs('top_images', exist_ok=True)
    for class_name, (top_images, top_scores) in baseline_top_images.items():
        for i, (img, score) in enumerate(zip(top_images, top_scores)):
            img = img.cpu().permute(1, 2, 0).numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.title(f"{class_name} - Score: {score:.4f}")
            plt.axis('off')
            plt.savefig(f'top_images/{class_name}_top_{i+1}.png')
            plt.close()

    # Augment dataset with GAN
    print("\n--- Augmenting Dataset with GAN ---")
    augmented_images, augmented_labels = augment_with_gan(train_dataset, num_samples_per_class=100)

    # Create augmented dataset
    augmented_dataset = AugmentedDataset(train_dataset, augmented_images, augmented_labels)
    
    # Create data loaders for augmented dataset
    augmented_train_loader = DataLoader(augmented_dataset, batch_size=BATCH_SIZE, shuffle=True)
    augmented_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train augmented VGGNet
    print("\n--- Training Augmented VGGNet ---")
    augmented_model = VGGClassifier().to(device)
    augmented_optimizer = optim.Adam(augmented_model.parameters(), lr=LR)

    # Training loop
    best_map = 0.0
    augmented_train_losses = []
    augmented_val_losses = []
    augmented_maps = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train for one epoch
        train_loss = train_epoch(augmented_model, augmented_train_loader, criterion, augmented_optimizer)
        
        # Evaluate on validation set
        val_loss, mean_ap, ap_scores = evaluate(augmented_model, augmented_val_loader, criterion)
        
        # Record statistics
        augmented_train_losses.append(train_loss)
        augmented_val_losses.append(val_loss)
        augmented_maps.append(mean_ap)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean AP: {mean_ap:.4f}")

        # Save best model
        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(augmented_model.state_dict(), 'best_augmented_model.pth')

    # Get top images for each class from augmented model
    print("\n--- Getting Top 10 Images per Class (Augmented) ---")
    augmented_top_images = {}
    for class_idx, class_name in enumerate(VOC_CLASSES):
        top_images, top_scores = get_top_images(augmented_model, val_loader, class_idx, top_n=10)
        augmented_top_images[class_name] = (top_images, top_scores)

    # Save top images
    os.makedirs('top_images', exist_ok=True)
    for class_name, (top_images, top_scores) in augmented_top_images.items():
        for i, (img, score) in enumerate(zip(top_images, top_scores)):
            img = img.cpu().permute(1, 2, 0).numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.title(f"{class_name} - Score: {score:.4f}")
            plt.axis('off')
            plt.savefig(f'top_images/{class_name}_top_{i+1}.png')
            plt.close()

    # Print final results
    print("\n--- Final Results ---")
    print(f"Best MAP (Baseline): {max(baseline_maps):.4f}")
    print(f"Best MAP (Augmented): {max(augmented_maps):.4f}")

    # Save results
    results = {
        'baseline_train_losses': baseline_train_losses,
        'baseline_val_losses': baseline_val_losses,
        'baseline_maps': baseline_maps,
        'augmented_train_losses': augmented_train_losses,
        'augmented_val_losses': augmented_val_losses,
        'augmented_maps': augmented_maps
    }
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n--- Experiment Completed ---")

if __name__ == "__main__":
    run_experiment()
