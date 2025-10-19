import os
import argparse
from PIL import Image
from PIL import Image as ImageOps # Alias para o filtro LANCZOS

TARGET_SIZE = (256, 256)

def processar_imagens(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Diretório de saída garantido: {output_dir}")

    for filename in os.listdir(input_dir):
        input_filepath = os.path.join(input_dir, filename)

        if not os.path.isfile(input_filepath):
            continue

        try:
            img = Image.open(input_filepath)
            
            print(f"Processando: {filename} ({img.size[0]}x{img.size[1]} - {img.format})")
            
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')

            resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            
            resized_img.save(output_filepath, "PNG")
            
            print(f"Salvo com sucesso como: {output_filename} ({resized_img.size[0]}x{resized_img.size[1]})")

        except Exception as e:
            print(f"ERRO ao processar {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Redimensiona imagens em um diretório para 256x256 e converte para PNG.')
    
    parser.add_argument('input_dir', 
                        type=str,
                        help='O caminho para o diretório contendo as imagens originais.')
    
    parser.add_argument('output_dir', 
                        type=str,
                        help='O caminho para o diretório onde as imagens processadas (PNG 256x256) serão salvas.')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"ERRO: O diretório de entrada '{args.input_dir}' não existe.")
        return

    processar_imagens(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()