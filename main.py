import os
import ultralytics
from ultralytics import YOLO

dir = 'datasets\\'

def main():
    ultralytics.checks()  # Realiza verificações básicas para garantir que o ambiente está configurado corretamente
    model = YOLO('yolov8m.pt')  # Carrega um modelo pré-treinado (recomendado para treinamento)

    # Treine o modelo usando o conjunto de dados especificado no arquivo config.yaml
    # epochs=200: Treina o modelo por 200 épocas
    # patience=100: Define a paciência do treinamento para 100 épocas (quantidade de épocas sem melhoria antes de parar)
    # val=True: Realiza validação ao final de cada época de treinamento
    # batch=-1: Ajusta automaticamente o tamanho do lote com base nos recursos disponíveis
    # single_cls=True: Trata todas as classes como uma única classe durante o treinamento (útil para detecção de uma única classe)
    results = model.train(data="config.yaml", epochs=200, patience=100, val=True, batch=1, single_cls=True)  

    # Avalia o desempenho do modelo no conjunto de validação
    # conf=0.50: Define um limiar de confiança de 50% para as predições durante a validação
    # save_json=True: Salva os resultados da avaliação em um arquivo JSON
    results = model.val(conf=0.01, save_json=True)  

    # Realiza a predição em uma imagem específica localizada no caminho 'datasets\\test\\store.jpg'
    # conf=0.50: Define um limiar de confiança de 50% para as predições
    results = model.predict(['datasets\\test\\store.jpg'], conf=0.4, show_labels=False)

    # Process results list
    for result in results:
    # Exibe o resultado na tela, mostrando apenas as caixas delimitadoras e a confiabilidade, sem as labels
        result.show(labels=False, conf=True)
        # Constrói o caminho completo para salvar o arquivo com o resultado
        save_path = os.path.join(dir + 'results', 'result_v3.jpg')
        result.save(filename=save_path)

if __name__ == '__main__':
    main()
