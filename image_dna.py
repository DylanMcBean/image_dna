import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as TF
import random
import uuid
import string
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import cv2
import base64
import math
from sklearn.model_selection import train_test_split

# =============================================================================
# Image Processing & Variation Generation
# =============================================================================

class ImageProcessor():
    def __init__(self, image_folder: str, num_variations=3, max_threads=4, save_folder='processed_variations', image_size=224):
        self.image_folder = image_folder
        self.image_paths = glob.glob(os.path.join(image_folder, '*.png'))
        self.image_paths = random.sample(self.image_paths, max(1000, len(self.image_paths)))
        self.num_variations = num_variations
        self.max_threads = max_threads
        self.save_folder = save_folder
        self.size = image_size

    def process_and_save_images(self):
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            list(tqdm(executor.map(self._process_and_save_image, self.image_paths),
                      total=len(self.image_paths),
                      desc="Processing and saving images"))

    def _process_and_save_image(self, image_path):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        
        image = Image.open(image_path).convert('RGB')
        image = self.fit_to_size(image, self.size)
        image_np = np.array(image)

        variations = [image_np]
        for _ in range(self.num_variations - 1):
            variations.append(self._create_variation(image_np))

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.save_folder, f"{image_id}.npy")
        np.save(save_path, np.array(variations, dtype=np.uint8))

    def fit_to_size(self, image: Image.Image, size=224) -> Image.Image:
        if size not in [512, 256, 224, 128, 64]:
            raise ValueError("Invalid size, must be one of [512,256,224,128,64]")
        orig_w, orig_h = image.size
        scale = size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_image = image.resize((new_w, new_h), Image.LANCZOS)

        new_image = Image.new("RGB", (size, size), (0, 0, 0))
        offset = ((size - new_w) // 2, (size - new_h) // 2)
        new_image.paste(resized_image, offset)
        return new_image

    def _create_variation(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        operations = [
            lambda img: img.rotate(90),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            lambda img: img.filter(ImageFilter.BLUR),
            lambda img: img.filter(ImageFilter.EMBOSS),
            lambda img: img.filter(ImageFilter.CONTOUR),
            lambda img: img.filter(ImageFilter.SHARPEN),
            lambda img: self.to_grayscale(img),
            lambda img: ImageOps.invert(img),
            lambda img: self.textOverlay(img),
            lambda img: self.obscureImage(img),
            lambda img: self.add_noise(img)
        ]

        for _ in range(4):
            if random.uniform(0, 1) < 0.5:
                operation = random.choice(operations)
                pil_image = operation(pil_image)
        return np.array(pil_image)

    def add_noise(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image).astype(np.float64)
        noise = np.random.normal(0, 255 * 0.05, image_np.shape)
        noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def obscureImage(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)
        h, w, _ = image_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        radius = min(h, w) // 4
        center_x = random.randint(radius, w - radius)
        center_y = random.randint(radius, h - radius)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        image_np[mask == 255] = 0
        return Image.fromarray(image_np)

    def textOverlay(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)
        h, w, _ = image_np.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        cv2.putText(image_np, text, (w // 4, h // 2), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        return Image.fromarray(image_np)

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        return image.convert('L').convert('RGB')

# =============================================================================
# Dataset Classes for Training and Testing
# =============================================================================

class ImagePairDataset(Dataset):
    def __init__(self, image_ids, num_pairs=3, save_folder='processed_variations'):
        self.image_ids = image_ids
        self.num_pairs = num_pairs
        self.save_folder = save_folder

    def __len__(self):
        return len(self.image_ids) * self.num_pairs * 2

    def __getitem__(self, idx):
        image_idx = idx // (self.num_pairs * 2)
        image_id1 = self.image_ids[image_idx]

        variations1 = np.load(os.path.join(self.save_folder, f"{image_id1}.npy"), allow_pickle=True)
        if idx % 2 == 0:
            image_id2 = image_id1
            variations2 = variations1
            label = 1
        else:
            other_ids = list(set(self.image_ids) - {image_id1})
            image_id2 = random.choice(other_ids) if other_ids else image_id1
            variations2 = np.load(os.path.join(self.save_folder, f"{image_id2}.npy"), allow_pickle=True)
            label = 0

        img1 = random.choice(variations1)
        img2 = random.choice(variations2)
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        label = torch.tensor(label, dtype=torch.float32)
        return img1, img2, label

class ImagePairTestDataset(Dataset):
    def __init__(self, image_ids, save_folder='processed_variations', fixed_seed=42):
        self.image_ids = image_ids
        self.save_folder = save_folder
        self.locked_pairs_info = self._generate_locked_pairs_info(fixed_seed)

    def _generate_locked_pairs_info(self, fixed_seed):
        rng = random.Random(fixed_seed)
        locked_info = []
        for image_id in self.image_ids:
            variations = np.load(os.path.join(self.save_folder, f"{image_id}.npy"), allow_pickle=True)
            num_variations = variations.shape[0]
            idx1 = rng.randint(0, num_variations - 1)
            idx2 = rng.randint(0, num_variations - 1)
            locked_info.append((image_id, image_id, idx1, idx2, 1))
            
            other_ids = list(set(self.image_ids) - {image_id})
            if other_ids:
                other_image_id = rng.choice(other_ids)
                variations_other = np.load(os.path.join(self.save_folder, f"{other_image_id}.npy"), allow_pickle=True)
                num_variations_other = variations_other.shape[0]
                idx1 = rng.randint(0, num_variations - 1)
                idx2 = rng.randint(0, num_variations_other - 1)
                locked_info.append((image_id, other_image_id, idx1, idx2, 0))
        return locked_info

    def __len__(self):
        return len(self.locked_pairs_info)

    def __getitem__(self, idx):
        image_id1, image_id2, idx1, idx2, label = self.locked_pairs_info[idx]
        variations1 = np.load(os.path.join(self.save_folder, f"{image_id1}.npy"), allow_pickle=True)
        variations2 = np.load(os.path.join(self.save_folder, f"{image_id2}.npy"), allow_pickle=True)
        img1 = variations1[idx1]
        img2 = variations2[idx2]
        t_img1 = TF.to_tensor(img1)
        t_img2 = TF.to_tensor(img2)
        t_label = torch.tensor(label, dtype=torch.float32)
        return t_img1, t_img2, t_label

# =============================================================================
# Model Components
# =============================================================================
class SimplifiedSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SimplifiedSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        reduced_H, reduced_W = max(H // 8, 1), max(W // 8, 1)
        x_reduced = torch.nn.functional.adaptive_avg_pool2d(x, (reduced_H, reduced_W))

        q = self.query(x_reduced).view(B, -1, reduced_H * reduced_W).permute(0, 2, 1)
        k = self.key(x_reduced).view(B, -1, reduced_H * reduced_W)
        v = self.value(x_reduced).view(B, -1, reduced_H * reduced_W)

        attn = torch.nn.functional.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, reduced_H, reduced_W)
        out = torch.nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return self.gamma * out + x

class ImageDNA(nn.Module):
    def __init__(self, feature_dim=64):
        super(ImageDNA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            SimplifiedSelfAttention(256),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DNAComparator(nn.Module):
    def __init__(self, dna_dim=64):
        super(DNAComparator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dna_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01),
            nn.Linear(64, 1)
        )

    def forward(self, dna1, dna2):
        x = torch.cat((dna1, dna2), dim=1)
        return self.classifier(x)

# =============================================================================
# Visualization and Utility Functions
# =============================================================================

def create_dna_image(dna_vector: np.ndarray) -> np.ndarray:
    dna_min, dna_max = dna_vector.min(), dna_vector.max()
    dna_norm = (dna_vector - dna_min) / (dna_max - dna_min + 1e-5) * 255
    dna_norm = dna_norm.astype(np.uint8)
    dna_img = dna_norm.reshape((8, 8))
    dna_img = cv2.resize(dna_img, (64, 64), interpolation=cv2.INTER_NEAREST)
    return dna_img

def pad_dna_image(dna_img: np.ndarray, target_height: int) -> np.ndarray:
    current_height, width = dna_img.shape
    pad_total = target_height - current_height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    padded = np.pad(dna_img, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    return padded_rgb

def display_similarity_scores(dataset, dna_model, comparator_model, device, epoch, loss):
    random_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
    fig, ax = plt.subplots(figsize=(20, 10))
    combined_title = ""
    images = []

    for idx in random_indices:
        img1, img2, similarity = dataset[idx]

        if not isinstance(img1, torch.Tensor):
            image1_tensor = TF.to_tensor(img1)
        else:
            image1_tensor = img1
        if not isinstance(img2, torch.Tensor):
            image2_tensor = TF.to_tensor(img2)
        else:
            image2_tensor = img2

        image1_tensor = image1_tensor.unsqueeze(0).to(device)
        image2_tensor = image2_tensor.unsqueeze(0).to(device)

        dna1 = dna_model(image1_tensor)
        dna2 = dna_model(image2_tensor)
        similarity_score = torch.sigmoid(comparator_model(dna1, dna2)).item()

        if similarity == 1:
            correct = similarity_score > 0.5
        else:
            correct = similarity_score <= 0.5

        if isinstance(img1, torch.Tensor):
            pil_img1 = TF.to_pil_image(img1).resize((256, 256), Image.LANCZOS)
        else:
            pil_img1 = TF.to_pil_image(TF.to_tensor(img1)).resize((256, 256), Image.LANCZOS)
        if isinstance(img2, torch.Tensor):
            pil_img2 = TF.to_pil_image(img2).resize((256, 256), Image.LANCZOS)
        else:
            pil_img2 = TF.to_pil_image(TF.to_tensor(img2)).resize((256, 256), Image.LANCZOS)

        left_img = np.array(pil_img1)
        right_img = np.array(pil_img2)

        dna1_vector = dna1.cpu().detach().numpy().squeeze()
        dna2_vector = dna2.cpu().detach().numpy().squeeze()
        dna1_img = create_dna_image(dna1_vector)
        dna2_img = create_dna_image(dna2_vector)

        dna1_padded = pad_dna_image(dna1_img, target_height=256)
        dna2_padded = pad_dna_image(dna2_img, target_height=256)

        combined_image = np.concatenate([dna1_padded, left_img, right_img, dna2_padded], axis=1)
        images.append(combined_image)

        combined_title += f"Similarity: {similarity_score:.4f} "
        if not correct:
            combined_title += r"INCORRECT" + "\n"
        else:
            combined_title += r"CORRECT" + "\n"

    gap = 20
    max_width = max(img.shape[1] for img in images)
    gap_image = np.zeros((gap, max_width, 3), dtype=np.uint8)
    final_image = np.concatenate([np.concatenate([img, gap_image], axis=0) for img in images], axis=0)

    ax.imshow(final_image)
    ax.set_title(f"Epoch: {epoch+1} / Loss: {loss:.5f}\n{combined_title.strip()}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"similarity_scores_epoch_{epoch+1}.png", bbox_inches='tight')
    plt.close()


def get_activation(model, layer_name, x):
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()
    hook = getattr(model, layer_name).register_forward_hook(hook_fn)
    _ = model(x)
    hook.remove()
    return activation

def plot_attention_grid(attn_maps, epoch, save_path):
    attn_maps = attn_maps.cpu().numpy()
    num_channels = attn_maps.shape[0]
    grid_size = math.ceil(math.sqrt(num_channels))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < num_channels:
            axes[i].imshow(attn_maps[i], cmap='viridis')
        axes[i].axis('off')
    fig.suptitle(f"Epoch {epoch+1}: Attention Maps")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def display_learned_values(dna_model, comparator_model, dataset, device, epoch, image1):
    dna_model.eval()
    comparator_model.eval()
    with torch.no_grad():
        feature_maps = dna_model.features[0](image1)
        feature_maps_np = feature_maps.squeeze(0).cpu().numpy()
        num_features = feature_maps_np.shape[0]
        grid_size = math.ceil(math.sqrt(num_features))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        axes = axes.flatten()
        for i in range(len(axes)):
            if i < num_features:
                axes[i].imshow(feature_maps_np[i], cmap='viridis')
            axes[i].axis('off')
        fig.suptitle(f"Epoch {epoch+1}: Feature Maps (First Conv Layer)")
        plt.tight_layout()
        plt.savefig(f"feature_maps_epoch_{epoch+1}.png", bbox_inches='tight')
        plt.close()

        attn_maps = dna_model.features[4](dna_model.features[3](
            dna_model.features[2](dna_model.features[1](dna_model.features[0](image1)))))
        attn_maps = attn_maps.squeeze(0)
        plot_attention_grid(attn_maps, epoch, f"attention_maps_epoch_{epoch+1}.png")

    kernels = dna_model.features[0].weight.data.cpu().numpy()
    min_w = np.min(kernels)
    max_w = np.max(kernels)
    kernels = (kernels - min_w) / (max_w - min_w)
    num_kernels = kernels.shape[0]
    grid_size = math.ceil(math.sqrt(num_kernels))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < num_kernels:
            kernel = kernels[i].transpose(1, 2, 0)
            ax.imshow(kernel)
        ax.axis('off')
    fig.suptitle(f"Epoch {epoch+1}: Kernels of First Conv Layer")
    plt.tight_layout()
    plt.savefig(f"kernels_epoch_{epoch+1}.png", bbox_inches='tight')
    plt.close()
    dna_model.train()
    comparator_model.train()

# =============================================================================
# Training and Evaluation
# =============================================================================

def evaluate_locked_pairs(test_dataset, dna_model, comparator_model, device, batch_size=32):
    dna_model.eval()
    comparator_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images1, images2, labels in test_loader:
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            dna1 = dna_model(images1)
            dna2 = dna_model(images2)
            outputs = comparator_model(dna1, dna2).squeeze(1)
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    outputs = torch.cat(all_outputs)
    labels = torch.cat(all_labels)
    
    loss = nn.BCEWithLogitsLoss()(outputs, labels).item()
    preds = (torch.sigmoid(outputs) > 0.5).float()
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    
    dna_model.train()
    comparator_model.train()
    return loss, accuracy


def train_model(dna_model, comparator_model, train_dataset, test_dataset, epochs=10, lr=0.001, device='cpu', batch_size=16, save_interval=5):
    dna_model.train()
    comparator_model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(dna_model.parameters()) + list(comparator_model.parameters()), lr=lr)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        tqdm.write(f"Loading latest checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        dna_model.load_state_dict(checkpoint['dna_model_state_dict'])
        comparator_model.load_state_dict(checkpoint['comparator_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        tqdm.write(f"Resuming training from epoch {start_epoch} with loss {best_loss:.4f}")
    else:
        start_epoch = 0
        best_loss = float('inf')
        tqdm.write("No checkpoint found. Starting training from scratch.")
    end_epoch = start_epoch + epochs
    epoch_progress = tqdm(range(start_epoch, end_epoch), desc="Training Progress", position=0, leave=True)
    for epoch in epoch_progress:
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        sample_image1, _, _ = random.choice(train_dataset)
        sample_image1 = sample_image1.unsqueeze(0).to(device)
        running_loss = 0.0
        batch_progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{end_epoch} - Batch Progress", position=1, leave=False)
        for images1, images2, labels in batch_progress:
            images1, images2 = images1.to(device), images2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            dna1 = dna_model(images1)
            dna2 = dna_model(images2)
            outputs = comparator_model(dna1, dna2).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_progress.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(dataloader)
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'dna_model_state_dict': dna_model.state_dict(),
                'comparator_model_state_dict': comparator_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")
        test_loss, test_accuracy = evaluate_locked_pairs(test_dataset, dna_model, comparator_model, device, batch_size=batch_size)
        tqdm.write(f"Epoch [{epoch+1}/{end_epoch}], Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}%")
        display_similarity_scores(test_dataset, dna_model, comparator_model, device, epoch, avg_loss)
        display_learned_values(dna_model, comparator_model, train_dataset, device, epoch, sample_image1)

# =============================================================================
# Main Execution: Split Images into Training and Testing Sets
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dataset_path = './data'
    image_processor = ImageProcessor(image_folder=dataset_path, num_variations=5, max_threads=8, image_size=224)
    if os.path.exists(dataset_path):
        print("Loading dataset from file")
    else:
        print("Generating dataset")
        image_processor.process_and_save_images()
    image_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(image_processor.save_folder, "*.npy"))]
    
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    train_dataset = ImagePairDataset(train_ids, num_pairs=10, save_folder=image_processor.save_folder)

    test_dataset = ImagePairTestDataset(test_ids, save_folder=image_processor.save_folder)
    
    dna_model = ImageDNA(feature_dim=64).to(device)
    comparator_model = DNAComparator(dna_dim=64).to(device)
    
    train_model(dna_model, comparator_model, train_dataset, test_dataset,
                epochs=50, lr=0.001, device=device, batch_size=128, save_interval=1)
