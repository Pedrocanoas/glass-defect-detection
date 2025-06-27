import os

# Caminho da pasta onde estão os arquivos
pasta = 'images'

# Lista de arquivos na pasta (ignorando pastas)
arquivos = [f for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]

# Ordena para ter consistência (opcional)
arquivos.sort()

for index, nome_arquivo in enumerate(arquivos):
    nome_completo_antigo = os.path.join(pasta, nome_arquivo)
    
    # Obtém a extensão
    _, extensao = os.path.splitext(nome_arquivo)
    
    # Novo nome
    novo_nome = f"image_{index}{extensao}"
    nome_completo_novo = os.path.join(pasta, novo_nome)
    
    # Renomeia
    os.rename(nome_completo_antigo, nome_completo_novo)

print("Arquivos renomeados com sucesso!")
