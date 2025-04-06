import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, resnet152, ResNet152_Weights, efficientnet_b0, EfficientNet_B0_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# torch.autograd.set_detect_anomaly(True)

class LatentMultiHeadAttention(nn.Module):
    """Latent Multi-Head Attention using fixed latent spatial resolution."""
    def __init__(self, dim=320, num_heads=5, latent_dim=49):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        # latent_tokens now has shape (1, latent_dim, dim) with latent_dim=49 (7x7 grid)
        self.latent_tokens = nn.Parameter(torch.randn(1, latent_dim, dim))

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape  # Expected input: (B, 320, H, W)
        # Flatten spatial dimensions: (B, HW, C)
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        
        # Expand latent tokens: (B, latent_dim, C)
        latent_expanded = self.latent_tokens.expand(B, -1, -1)  # (B, latent_dim, C)
        
        # Linear projections
        q = self.query(latent_expanded)  # (B, latent_dim, C)
        k = self.key(x)                  # (B, HW, C)
        v = self.value(x)                # (B, HW, C)
        
        # Reshape to multi-head attention
        # Split heads: (B, num_heads, seq_len, head_dim)
        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, latent_dim, head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, HW, head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, HW, head_dim)
        
        # Compute attention for each head
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, num_heads, latent_dim, HW)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, latent_dim, head_dim)
        
        # Merge heads: (B, latent_dim, C)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, C)
        
        # Final projection
        out = self.proj(out)  # (B, latent_dim, C)
        
        # Reshape to spatial layout: (B, C, new_h, new_w)
        out = out.permute(0, 2, 1)  # (B, C, latent_dim)
        new_spatial = int(self.latent_tokens.shape[1] ** 0.5)  # should be 7
        out = out.view(B, C, new_spatial, new_spatial)  # (B, C, 7, 7)
        
        return out

class MultiScaleMobileNetDehaze(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = T.Resize((256, 256))  # Keep original input size

        # Get MobileNetV3-Small model
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Extract features at different scales (keep original)
        # Scale 1: 1/4 resolution (64x64) - 16 channels
        self.encoder_s1 = nn.Sequential(*list(mobilenet.features[:2]))
        
        # Scale 2: 1/8 resolution (32x32) - 24 channels
        self.encoder_s2 = nn.Sequential(*list(mobilenet.features[2:4]))
        
        # Scale 3: 1/16 resolution (16x16) - 40 channels
        self.encoder_s3 = nn.Sequential(*list(mobilenet.features[4:7]))
        
        # Channel mappers for each scale - slightly reduced
        self.intermediate_dim = 320  # Down from 384
        self.mapper_s1 = nn.Conv2d(16, self.intermediate_dim // 4, kernel_size=1)
        self.mapper_s2 = nn.Conv2d(24, self.intermediate_dim // 4, kernel_size=1)
        self.mapper_s3 = nn.Conv2d(40, self.intermediate_dim // 2, kernel_size=1)
        
        # Multi-head attention module
        self.mhla = LatentMultiHeadAttention(dim=self.intermediate_dim, num_heads=5, latent_dim=49)
        
        # Feature fusion layer
        self.fusion = nn.Conv2d(self.intermediate_dim, self.intermediate_dim, kernel_size=3, padding=1)
        
        # Decoder: Slightly reduced dimensions
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(self.intermediate_dim, 160, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(160, 80, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder_3 = nn.Conv2d(80, 3, kernel_size=3, padding=1)

        # Skip connection mapping layers
        # self.skip_mapper_1 = nn.Conv2d(3, 160, kernel_size=1)
        # self.skip_mapper_2 = nn.Conv2d(3, 80, kernel_size=1)
        self.skip_mapper_1 = nn.Conv2d(80, 160, kernel_size=1)  # For m1 features
        self.skip_mapper_2 = nn.Conv2d(80, 80, kernel_size=1)   # For m2 features

    def forward(self, x):
        original_size = (x.shape[2], x.shape[3])
        input_x = x.clone()  # Store input for skip connections
        x = self.downsample(x)          # (B, 3, 256, 256)
        input_resized = self.downsample(input_x)  # Resize input for skip connections
        
        # Multi-scale feature extraction
        f1 = self.encoder_s1(x)         # (B, 16, 64, 64)
        f2 = self.encoder_s2(f1)        # (B, 24, 32, 32)
        f3 = self.encoder_s3(f2)        # (B, 40, 16, 16)
        
        # Transform features to common dimension
        m1 = self.mapper_s1(f1)         # (B, 80, 64, 64)
        m2 = self.mapper_s2(f2)         # (B, 80, 32, 32)
        m3 = self.mapper_s3(f3)         # (B, 160, 16, 16)
        
        # Resize all features to the same resolution (16x16)
        m1 = F.interpolate(m1, size=(16, 16), mode='bilinear', align_corners=False)
        m2 = F.interpolate(m2, size=(16, 16), mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([m1, m2, m3], dim=1)  # (B, 320, 16, 16)
        
        # Apply fusion layer
        fused_features = self.fusion(multi_scale_features)
        
        # Apply attention
        attended_features = self.mhla(fused_features)  # (B, 320, 7, 7)
        
        # Upsample to original resolution
        x = F.interpolate(attended_features, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Decode with skip connections
        x1 = self.decoder_1(x)             # (B, 160, 256, 256)
        skip1 = self.skip_mapper_1(F.interpolate(m1, size=(256, 256), mode='bilinear', align_corners=False))   # Map from 80->160 channels
        x1 = x1 + skip1  # First skip connection
        
        x2 = self.decoder_2(x1)            # (B, 80, 256, 256)
        skip2 = self.skip_mapper_2(F.interpolate(m2, size=(256, 256), mode='bilinear', align_corners=False))  # Map from 80->80 channels
        x2 = x2 + skip2  # Second skip connection
        
        x3 = self.decoder_3(x2)            # (B, 3, 256, 256)
        x3 = x3 + input_resized  # Direct skip connection to input (residual learning)
        
        x_out = torch.clamp(x3, 0.0, 1.0)
        
        # Resize back to original input size if needed
        if original_size != (256, 256):
            x_out = F.interpolate(x_out, size=original_size, mode='bilinear', align_corners=False)
            
        return x_out

# Model Initialization
model = MultiScaleMobileNetDehaze().to(device)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

# Testing with a dummy input
dummy_input = torch.randn(1, 3, 256, 256).to(device)
output = model(dummy_input)
print("Output shape:", output.shape)  # Expected: [1, 3, 256, 256]

input_shape = (1, 3, 256, 256)
summary(model, input_size=input_shape, col_names=["input_size", "output_size", "num_params"])

class DehazeDataset(Dataset):
    def __init__(self, haze_dir, clear_dir, transform=None):
        self.haze_dir = haze_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.image_ids = sorted(set(f.split('_')[0] for f in os.listdir(haze_dir)))
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        haze_files = sorted([f for f in os.listdir(self.haze_dir) if f.startswith(image_id)])
        clear_file = f"{image_id}.png"
        
        if not haze_files or not os.path.exists(os.path.join(self.clear_dir, clear_file)):
            return None
        
        haze_path = os.path.join(self.haze_dir, haze_files[0])  # Take first hazy image
        clear_path = os.path.join(self.clear_dir, clear_file)
        
        haze_img = cv2.imread(haze_path)
        clear_img = cv2.imread(clear_path)
        
        haze_img = cv2.cvtColor(haze_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            haze_img = self.transform(haze_img)
            clear_img = self.transform(clear_img)
        
        return haze_img, clear_img

def train_model():
    # Use more efficient transforms
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    # Create datasets with train/val/test splits
    full_dataset = DehazeDataset("ITS_haze", "ITS_clear", transform=transform)
    
    # Split dataset: 70% train, 15% validation, 15% test
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create data loaders for each split
    train_loader = DataLoader(
        train_dataset, 
        batch_size=10,
        shuffle=True, 
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=10,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=10,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"Length of train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    print(f"Number of batches in train_loader: {len(train_loader)}, val_loader: {len(val_loader)}, test_loader: {len(test_loader)}")

    # Initialize optimized model
    model = MultiScaleMobileNetDehaze().to(device)
    
    # Use mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # Improved LR scheduler with correct steps calculation
    total_steps = 100 * len(train_loader)  # Updated to use train_loader only
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000
    )

    # Use more efficient loss and metrics
    loss_fn = nn.SmoothL1Loss()
    
    # Use smaller EfficientNet instead of ResNet152 for perceptual loss
    feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features[:4].eval().to(device)
    for param in feature_extractor.parameters():
        param.requires_grad = False
        
    # Perceptual loss with feature caching
    def perceptual_loss(pred, target):
        with torch.no_grad():
            target_feats = feature_extractor(target)
        pred_feats = feature_extractor(pred)
        return F.l1_loss(pred_feats, target_feats)

    # Use torchmetrics for efficient GPU-based metrics
    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure().to(device)

    # Initialize TensorBoard
    writer = SummaryWriter("runs/optimized_dehazing")

    num_epochs = 50
    best_val_ssim = 0.0
    history = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Early stopping parameters
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        
        # Reset metrics for training
        psnr_metric.reset()
        ssim_metric.reset()
        
        for hazy, clear in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            hazy, clear = hazy.to(device), clear.to(device)
            
            # Use mixed precision where available
            if scaler is not None:
                with torch.autocast("cuda", enabled=True):
                    output = model(hazy)
                    loss_a = loss_fn(output, clear)
                    loss_b = perceptual_loss(output, clear)
                    loss = loss_a + 0.01 * loss_b
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                
                # Add gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(hazy)
                loss_a = loss_fn(output, clear)
                loss_b = perceptual_loss(output, clear)
                loss = loss_a + 0.01 * loss_b
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            train_loss += loss.item()
            
            # Update metrics on GPU
            psnr_metric.update(output, clear)
            ssim_metric.update(output, clear)
            
            # Step the scheduler per batch
            scheduler.step()
        
        # Compute average training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = psnr_metric.compute().item()
        avg_train_ssim = ssim_metric.compute().item()
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        
        # Reset metrics for validation
        psnr_metric.reset()
        ssim_metric.reset()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for hazy, clear in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}"):
                hazy, clear = hazy.to(device), clear.to(device)
                output = model(hazy)
                
                loss_a = loss_fn(output, clear)
                loss_b = perceptual_loss(output, clear)
                loss = loss_a + 0.01 * loss_b
                
                val_loss += loss.item()
                
                # Update metrics
                psnr_metric.update(output, clear)
                ssim_metric.update(output, clear)
        
        # Compute average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = psnr_metric.compute().item()
        avg_val_ssim = ssim_metric.compute().item()
        
        # Print epoch results
        print(f"Epoch {epoch+1}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.2f}, SSIM: {avg_train_ssim:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}")
        
        # Log to TensorBoard
        writer.add_scalars("Loss", {
            "train": avg_train_loss,
            "val": avg_val_loss
        }, epoch)
        writer.add_scalars("PSNR", {
            "train": avg_train_psnr,
            "val": avg_val_psnr
        }, epoch)
        writer.add_scalars("SSIM", {
            "train": avg_train_ssim,
            "val": avg_val_ssim
        }, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # Save training history
        history["train"].append({
            "epoch": epoch+1, 
            "loss": avg_train_loss, 
            "psnr": avg_train_psnr, 
            "ssim": avg_train_ssim, 
            "lr": optimizer.param_groups[0]['lr']
        })
        
        history["val"].append({
            "epoch": epoch+1, 
            "loss": avg_val_loss, 
            "psnr": avg_val_psnr, 
            "ssim": avg_val_ssim
        })
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_psnr": avg_train_psnr,
                "val_psnr": avg_val_psnr,
                "train_ssim": avg_train_ssim,
                "val_ssim": avg_val_ssim,
                "history": history
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save the best model based on validation SSIM
        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_psnr": avg_train_psnr,
                "val_psnr": avg_val_psnr,
                "train_ssim": avg_train_ssim,
                "val_ssim": avg_val_ssim
            }, "best_model.pth")
            print(f"New best model saved at epoch {epoch+1} with validation SSIM {best_val_ssim:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # --- TESTING PHASE ---
    print("\nEvaluating best model on test set...")
    
    # Load the best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_loss = 0.0
    
    # Reset metrics for testing
    psnr_metric.reset()
    ssim_metric.reset()
    
    with torch.no_grad():
        for hazy, clear in tqdm(test_loader, desc="Testing"):
            hazy, clear = hazy.to(device), clear.to(device)
            output = model(hazy)
            
            loss_a = loss_fn(output, clear)
            loss_b = perceptual_loss(output, clear)
            loss = loss_a + 0.01 * loss_b
            
            test_loss += loss.item()
            
            # Update metrics
            psnr_metric.update(output, clear)
            ssim_metric.update(output, clear)
    
    # Compute average test metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_test_psnr = psnr_metric.compute().item()
    avg_test_ssim = ssim_metric.compute().item()
    
    print(f"\nTest Results - Loss: {avg_test_loss:.4f}, PSNR: {avg_test_psnr:.2f}, SSIM: {avg_test_ssim:.4f}")
    
    # Save test results
    history["test"] = {
        "loss": avg_test_loss,
        "psnr": avg_test_psnr,
        "ssim": avg_test_ssim
    }
    
    # Save final results including test metrics
    final_results = {
        "best_epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "test_loss": avg_test_loss,
        "train_psnr": checkpoint["train_psnr"],
        "val_psnr": checkpoint["val_psnr"],
        "test_psnr": avg_test_psnr,
        "train_ssim": checkpoint["train_ssim"],
        "val_ssim": checkpoint["val_ssim"],
        "test_ssim": avg_test_ssim,
        "history": history
    }
    
    torch.save(final_results, "final_results.pth")
    
    writer.close()
    return model, history

# Entry point
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, history = train_model()
