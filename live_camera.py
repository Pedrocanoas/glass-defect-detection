import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

MODEL_PATH = 'best.pt'  # Caminho para o modelo YOLOv8
CAMERA_URL = 'http://10.109.61.34:8080/video'
INFERENCE_INTERVAL = 1
OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# Controle
last_inference_time = 0
running = True
annotated_frame = None
frame_for_inference = None
lock = threading.Lock()

# Limpa a pasta antes de começar
for filename in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Thread para inferência
def inference_thread():
    global frame_for_inference, annotated_frame, running

    while running:
        with lock:
            if frame_for_inference is not None:
                frame = frame_for_inference.copy()
                frame_for_inference = None
            else:
                frame = None

        if frame is not None:
            results = model.predict(source=frame, conf=1.0, verbose=False)
            annotated = results[0].plot()

            
            # Aumentar saturação
            annotated = aumentar_saturacao(annotated, fator=0.5)

            # Salva imagem anotada
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"inferencia_{timestamp}.jpg")
            cv2.imwrite(save_path, annotated)

            with lock:
                annotated_frame = annotated

        time.sleep(0.1)

def aumentar_saturacao(imagem, fator=1.5):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = s * fator
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    imagem_bgr = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imagem_bgr


# Conecta à câmera
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print("❌ Erro ao acessar a câmera.")
    exit()

# Inicia thread de inferência
thread = threading.Thread(target=inference_thread)
thread.start()

print("✅ Câmera conectada. Pressione 'q' para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Falha ao capturar frame.")
            break

        # Envia frame para inferência a cada intervalo
        if time.time() - last_inference_time > INFERENCE_INTERVAL:
            with lock:
                frame_for_inference = frame.copy()
            last_inference_time = time.time()

        # Mostra apenas a inferência (sem vídeo ao vivo)
        with lock:
            if annotated_frame is not None:
                cv2.imshow("📸 Inferência", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrompido.")

finally:
    running = False
    thread.join()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Finalizado.")
