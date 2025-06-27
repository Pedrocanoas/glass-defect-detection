import os
import time
from datetime import datetime
from ultralytics import YOLO
import ultralytics

# Configurações
DIR = 'datasets'
MODEL_PATH = 'yolov8m.pt'
DATA_YAML = 'config.yaml'
EPOCHS = 200
PATIENCE = 100
CONF_TRAIN = 0.01
CONF_PREDICT = 0.4
SINGLE_CLASS = True
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"result_{timestamp}.jpg"

RESULT_PATH = os.path.join(DIR, 'results', filename)
IMAGE_TO_PREDICT = os.path.join(DIR, 'images', 'test', 'store.jpg')

def main():
    start_time = time.time()

    print("🔍 Verificando ambiente...")
    ultralytics.checks()

    print(f"📦 Carregando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("🚀 Iniciando treinamento...")
    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        val=True,
        batch=1,
        single_cls=SINGLE_CLASS
    )

    print("📈 Avaliando modelo...")
    val_results = model.val(conf=CONF_TRAIN, save_json=True)

    print(f"🖼️ Realizando predição em: {IMAGE_TO_PREDICT}")
    pred_results = model.predict([IMAGE_TO_PREDICT], conf=CONF_PREDICT, show_labels=False)

    # Garante que a pasta de resultados existe
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    for result in pred_results:
        result.show(labels=False, conf=True)
        result.save(filename=RESULT_PATH)
        print(f"💾 Resultado salvo em: {RESULT_PATH}")

    print(f"✅ Finalizado em {time.time() - start_time:.2f} segundos.")

if __name__ == '__main__':
    main()
