import os
import time
from datetime import datetime
from ultralytics import YOLO
import ultralytics

# Configurações
DIR = 'dataset_tiled'
MODEL_PATH = 'yolov8m.pt'
DATA_YAML = 'config.yaml'
EPOCHS = 50
PATIENCE = 10
CONF_TRAIN = 0.01
CONF_PREDICT = 0.4
SINGLE_CLASS = True
IMAGE_TO_PREDICT_DIR = os.path.join(DIR, 'images', 'test')
RESULT_DIR = os.path.join(DIR, 'results')

def main():
    start_time = time.time()

    print("🔍 Verificando ambiente...")
    ultralytics.checks()

    print(f"📦 Carregando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("🚀 Iniciando treinamento...")
    train_results = model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=EPOCHS,
        patience=PATIENCE,
        val=True,
        cache=True,
        batch=2,
        single_cls=SINGLE_CLASS,
        lr0=0.002,
    )

    print("📈 Avaliando modelo...")
    val_results = model.val(conf=CONF_TRAIN, save_json=True)

    # Coleta todas as imagens do diretório de teste
    print(f"🖼️ Buscando imagens em: {IMAGE_TO_PREDICT_DIR}")
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(IMAGE_TO_PREDICT_DIR, f)
        for f in os.listdir(IMAGE_TO_PREDICT_DIR)
        if f.lower().endswith(image_extensions)
    ]

    if not image_paths:
        print(f"❌ Nenhuma imagem encontrada em {IMAGE_TO_PREDICT_DIR}")
        return

    print(f"🔍 {len(image_paths)} imagens encontradas. Iniciando predição...")
    pred_results = model.predict(source=image_paths, conf=CONF_PREDICT)

    # Garante que a pasta de resultados existe
    os.makedirs(RESULT_DIR, exist_ok=True)

    for i, result in enumerate(pred_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{i+1}_{timestamp}.jpg"
        result_path = os.path.join(RESULT_DIR, filename)
        result.show(labels=False, conf=True)
        result.save(filename=result_path)
        print(f"💾 Resultado salvo em: {result_path}")

    print(f"✅ Finalizado em {time.time() - start_time:.2f} segundos.")

if __name__ == '__main__':
    main()
