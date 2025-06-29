import os
import cv2

# CONFIGURAÇÕES
TILE_SIZE = 640
OVERLAP = 0  # Pode usar 0 ou algo como 64 se quiser tiles com sobreposição
CLASSES = None  # None = aceita qualquer classe

# Diretórios base
BASE_DIR = 'dataset'
OUT_BASE_DIR = 'dataset_tiled'
SPLITS = ['train', 'val', 'test']
IMG_DIR = os.path.join(BASE_DIR, 'images')
LBL_DIR = os.path.join(BASE_DIR, 'labels')
OUT_IMG_DIR = os.path.join(OUT_BASE_DIR, 'images_tiled')
OUT_LBL_DIR = os.path.join(OUT_BASE_DIR, 'labels_tiled')

def yolo_to_bbox(label, img_w, img_h):
    cls, x, y, w, h = map(float, label.split())
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return int(cls), x1, y1, x2, y2

def bbox_to_yolo(cls, x1, y1, x2, y2, tile_w, tile_h):
    x = ((x1 + x2) / 2) / tile_w
    y = ((y1 + y2) / 2) / tile_h
    w = (x2 - x1) / tile_w
    h = (y2 - y1) / tile_h
    return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

def process_split(split):
    os.makedirs(os.path.join(OUT_IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUT_LBL_DIR, split), exist_ok=True)

    images = sorted(os.listdir(os.path.join(IMG_DIR, split)))

    for img_file in images:
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(IMG_DIR, split, img_file)
        lbl_path = os.path.join(LBL_DIR, split, os.path.splitext(img_file)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Carrega labels
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]

        count = 0
        for y in range(0, img_h, TILE_SIZE - OVERLAP):
            for x in range(0, img_w, TILE_SIZE - OVERLAP):
                tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]
                if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                    continue

                tile_labels = []
                for label in labels:
                    cls, x1, y1, x2, y2 = yolo_to_bbox(label, img_w, img_h)
                    if x1 >= x + TILE_SIZE or x2 <= x or y1 >= y + TILE_SIZE or y2 <= y:
                        continue  # completamente fora do tile

                    # Calcula coordenadas relativas ao tile
                    nx1 = max(x1 - x, 0)
                    ny1 = max(y1 - y, 0)
                    nx2 = min(x2 - x, TILE_SIZE)
                    ny2 = min(y2 - y, TILE_SIZE)

                    if (nx2 - nx1) <= 1 or (ny2 - ny1) <= 1:
                        continue

                    yolo_label = bbox_to_yolo(cls, nx1, ny1, nx2, ny2, TILE_SIZE, TILE_SIZE)
                    tile_labels.append(yolo_label)

                if len(tile_labels) == 0:
                    continue

                tile_img_name = f"{os.path.splitext(img_file)[0]}_{count}.jpg"
                tile_lbl_name = f"{os.path.splitext(img_file)[0]}_{count}.txt"

                cv2.imwrite(os.path.join(OUT_IMG_DIR, split, tile_img_name), tile)
                with open(os.path.join(OUT_LBL_DIR, split, tile_lbl_name), 'w') as f:
                    f.write('\n'.join(tile_labels))

                count += 1

    print(f"✅ Split '{split}' processado com sucesso!")

for split in SPLITS:
    process_split(split)
