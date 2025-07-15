import os
import shutil
import random

# Configurações
input_dir = "rawdata"
output_dir = "dataset_rawdata1"
split_ratios = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

# Coleta todos os arquivos de imagem
image_dir = os.path.join(input_dir, "images")
label_dir = os.path.join(input_dir, "labels")

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Embaralhar para garantir aleatoriedade
random.shuffle(image_files)

# Calcular quantidades
total = len(image_files)
train_count = int(split_ratios["train"] * total)
val_count = int(split_ratios["val"] * total)

splits = {
    "train": image_files[:train_count],
    "val": image_files[train_count:train_count + val_count],
    "test": image_files[train_count + val_count:]
}

# Criar estrutura de pastas
for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Copiar arquivos
for split, files in splits.items():
    for img_file in files:
        base_name, _ = os.path.splitext(img_file)
        label_file = base_name + ".txt"

        # Caminhos origem
        img_src = os.path.join(image_dir, img_file)
        lbl_src = os.path.join(label_dir, label_file)

        # Caminhos destino
        img_dst = os.path.join(output_dir, "images", split, img_file)
        lbl_dst = os.path.join(output_dir, "labels", split, label_file)

        shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
        else:
            print(f"⚠️  Label não encontrada para: {img_file}")

print("✅ Dados organizados com sucesso!")
