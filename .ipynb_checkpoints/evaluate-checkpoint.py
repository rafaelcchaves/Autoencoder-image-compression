import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

INPUT_DIR = './images/input'        
LATENT_DIR = './images/compressed'      
OUTPUT_DIR = './images/output'
IMAGE_SIZE = (256, 256)               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # --- Encoder ---
        # Entrada: (B, 3, 256, 256)
        self.encoder = nn.Sequential(
            # 1ª Camada: (B, 3, 256, 256) -> (B, 16, 128, 128)
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 2ª Camada: (B, 16, 128, 128) -> (B, 32, 64, 64)
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 3ª Camada: (B, 32, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 4ª Camada: (B, 64, 32, 32) -> (B, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 5ª Camada: (B, 128, 16, 16) -> (B, 128, 8, 8)
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
            # Fim do Encoder. Saída é o espaço latente (128, 8, 8)
        )
        
        # --- Decoder ---
        # Entrada: (B, 128, 8, 8)
        self.decoder = nn.Sequential(
            # 1ª Camada: (B, 128, 8, 8) -> (B, 128, 16, 16)
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 2ª Camada: (B, 128, 16, 16) -> (B, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 3ª Camada: (B, 64, 32, 32) -> (B, 32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 4ª Camada: (B, 32, 64, 64) -> (B, 16, 128, 128)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 5ª Camada: (B, 16, 128, 128) -> (B, 3, 256, 256)
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Saída da imagem normalizada entre [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent_vector = encoded 
        decoded = self.decoder(latent_vector)
        return decoded, latent_vector

def calculate_psnr(original, reconstructed):
    orig_np = original.cpu().detach().numpy().transpose(1, 2, 0)
    recon_np = reconstructed.cpu().detach().numpy().transpose(1, 2, 0)
    return psnr_metric(orig_np, recon_np, data_range=1.0)

def calculate_ssim(original, reconstructed):
    orig_np = original.cpu().detach().numpy().transpose(1, 2, 0)
    recon_np = reconstructed.cpu().detach().numpy().transpose(1, 2, 0)
    return ssim_metric(orig_np, recon_np, data_range=1.0, channel_axis=-1)

def run_autoencoder_processing():
    """Main function to initialize, process, save, and calculate metrics."""
    
    for d in [INPUT_DIR, LATENT_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
    
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.bmp'))]
    if not image_files:
        print(f"ERROR: No images found in {INPUT_DIR}. Please add images to run the script.")
        return

    model = torch.load('models/model-v1', weights_only=False).eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    
    all_metrics = []
    print(f"Processing {len(image_files)} images on device: {DEVICE}")

    for filename in image_files:
        input_path = os.path.join(INPUT_DIR, filename)
        
        try:
            original_img_pil = Image.open(input_path).convert("RGB")
            original_tensor = transform(original_img_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                reconstructed_tensor, latent_tensor = model(original_tensor)

            original_single = original_tensor.squeeze(0)
            reconstructed_single = reconstructed_tensor.squeeze(0)
            
            output_path = os.path.join(OUTPUT_DIR, f'reconstructed_{filename}')
            save_image(reconstructed_single, output_path)
            
            latent_np = latent_tensor.cpu().squeeze(0).flatten().numpy()
            latent_path = os.path.join(LATENT_DIR, f'latent_{os.path.splitext(filename)[0]}.npy')
            np.save(latent_path, latent_np)
            
            psnr_val = calculate_psnr(original_single, reconstructed_single)
            ssim_val = calculate_ssim(original_single, reconstructed_single)
            
            all_metrics.append({'psnr': psnr_val, 'ssim': ssim_val})
            
            print(f"| {filename:<20} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if all_metrics:
        avg_psnr = np.mean([m['psnr'] for m in all_metrics])
        avg_ssim = np.mean([m['ssim'] for m in all_metrics])
        print("\n" + "="*50)
        print(f"SUMMARY: Processed {len(all_metrics)} images.")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("="*50)

if __name__ == '__main__':
    run_autoencoder_processing()