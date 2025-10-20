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


class AutoEncoderMaxPool(nn.Module):
    def __init__(self):
        super(AutoEncoderMaxPool, self).__init__()
        
        # --- Encoder ---
        # Usamos Conv2d(k=3, s=1, p=1) para aprender features sem alterar o tamanho,
        # e MaxPool2d(k=2, s=2) para reduzir (downsample) H e W por 2.
        
        # Entrada: (B, 3, 256, 256)
        self.encoder = nn.Sequential(
            # Bloco 1: 256x256 -> 128x128
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 16, 128, 128)

            # Bloco 2: 128x128 -> 64x64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, 64, 64)
            
            # Bloco 3: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, 32, 32)
            
            # Bloco 4: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 128, 16, 16)
            
            # Bloco 5: 16x16 -> 8x8
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (B, 128, 8, 8)
            
            # Fim do Encoder. Saída é o espaço latente (128, 8, 8)
        )
        
        # --- Decoder ---
        # Usamos Upsample(scale_factor=2) para dobrar H e W (operação oposta ao MaxPool)
        # e Conv2d(k=3, s=1, p=1) para refinar as features e ajustar os canais.
        
        # Entrada: (B, 128, 8, 8)
        self.decoder = nn.Sequential(
            # Bloco 1: 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 128, 16, 16)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            # Bloco 2: 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 128, 32, 32)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            # Bloco 3: 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 64, 64, 64)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            # Bloco 4: 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 32, 128, 128)
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            # Bloco 5: 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='nearest'), # (B, 16, 256, 256)
            # Camada final de convolução para ajustar os canais para 3 (RGB)
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            
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