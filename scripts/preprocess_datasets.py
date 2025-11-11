#!/usr/bin/env python3
"""
Preprocessamento dos datasets PKLot e Fruits360 para uso com LibTorch (C++).
Autor: Lucas Araujo
Descrição:
    Este script gera tensores (train/test) e metadados (.pt) que podem ser carregados
    diretamente no LibTorch para experimentos sobre precisão numérica (FP64, FP32, FP16, BF16).

Entradas:
    - Diretórios originais dos datasets.
Saídas:
    - /processed/<dataset>/{train_images.pt, train_labels.pt, test_images.pt, test_labels.pt}
"""

import gc
import os
import torch
import numpy as np
import threading
import time
from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

# ==============================
# 🔧 CONFIGURAÇÕES DO SCRIPT
# ==============================

# DATASET_NAME = "PKLot"
DATASET_NAME = "Fruits360"

if DATASET_NAME not in ["PKLot", "Fruits360"]:
    raise ValueError("DATASET_NAME deve ser 'PKLot' ou 'Fruits360'.")

# Trata a posicao do diretorio 'datasets'
# Busca no diretorio atual e no antessessor
# Se nao encontrar, lanca uma excecao
if not os.path.exists("./datasets"):
    if os.path.exists("../datasets"):
        os.chdir("..")
    else:
        raise FileNotFoundError("Diretório 'datasets' não encontrado.")

INPUT_DIR = f"./datasets/{DATASET_NAME}/raw"
OUTPUT_DIR = f"./datasets/{DATASET_NAME}/processed"
TEMP_DIR = f"./temp" # Diretório temporário para salvar tensores parciais

# Tamanho da imagem (redimensionamento padrão)
IMAGE_SIZE = (64, 64)

# Normalização (ImageNet padrão)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD =  [0.229, 0.224, 0.225]

# Percentual de treino/teste
TRAIN_RATIO = 0.8

# Seed para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

VERBOSE = True

NUM_THREADS = 2

TRANSFORM_COMPOSE = v2.Compose([
    v2.Resize(IMAGE_SIZE),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORM_MEAN, std=NORM_STD)  # Normalização ImageNet
])

has_clean = False

# ==============================
# 🧩 FUNÇÕES AUXILIARES
# ==============================

def term_width():
    """Retorna a largura do terminal."""
    return os.get_terminal_size().columns - 1

# Apenas imprime o verbose_text se VERBOSE for True
# Verbose text tem que estar antes da barra, ou seja, se for imprimir o texto, entao deve ser apagado toda a barra anterior e feito uma nova
# Sempre tem que ter um espaco em branco antes da barra
# Se VERBOSE e o verbose_text for vazio, nao imprime nada apenas a barra de progresso
# A barra precisa ser atualizada na mesma linha e ter as informacoes de progresso alem de uma informacao visual e de porcentual
# verbose_text sempre deve ser impresso antes da barra
def progress_bar(current, total, verbose_text="", finished=False):
    """Exibe uma barra de progresso no terminal."""
    global has_clean
    
    RETURN_TOKEN = '\r'
    LINE_ABOVE_TOKEN = '\033[F'
    BLANK_LINE_TOKEN = ' ' * (term_width() + 1) + '\n'
    
    if not VERBOSE:
        return

    fraction = current / total
    percent = fraction * 100
    percent_str = f" ({percent:6.2f}%)"
    progress_str = f"[{current}/{total}] "

    bar_length = term_width() - len(progress_str) - len(percent_str) - 3

    arrow = int(fraction * bar_length) * '█'
    spaces = (bar_length - len(arrow)) * ' '
    
    final_str = ""

    if verbose_text:
        # return_to_start_of_line()
        # goto_line_above()
        final_str = RETURN_TOKEN
        if has_clean:
            final_str += LINE_ABOVE_TOKEN
        final_str += f"{verbose_text}\n"

    bar_str = ""
    bar_str += RETURN_TOKEN
    bar_str += BLANK_LINE_TOKEN
    bar_str += f"{progress_str}[{arrow}{spaces}]{percent_str}"
    
    has_clean = True
    
    final_str += bar_str

    if finished:
        has_clean = False
        final_str += "\n"  # Nova linha ao finalizar

    print(final_str, end='')

def create_new_filename(dst_file):
    """Cria um novo nome de arquivo se o destino já existir."""
    base, ext = os.path.splitext(dst_file)
    index = 1
    new_file = f"{base}_{index}{ext}"
    while os.path.exists(new_file):
        index += 1
        new_file = f"{base}_{index}{ext}"
    return new_file

def increment_fruits_file_index(dst_file):
    """Incrementa o indice do arquivo para eviter sobrescrita no Fruits360.
        Para o dataset atual (100x100), o padrao eh:
           <rx_>index_100.jpg
        Onde: 
            r = rotacao (0, 1 ou 2)
            x = indice da imagem na classe
    """
    dst_dir, filename = os.path.split(dst_file)
    base, ext = os.path.splitext(filename)
    parts = base.split('_')
    if len(parts) < 2:
        return create_new_filename(dst_file)  # Formato inesperado, cria novo nome

    index_part = parts[-2]           # Parte do indice
    size_part = parts[-1]            # Parte do tamanho (100)
    
    try:
        index = int(index_part)
    except ValueError:
        return create_new_filename(dst_file)  # Formato inesperado, cria novo nome

    index += 1  # Incrementa o indice

    if len(parts) == 3:
        rotation_part = parts[0]     # Parte da rotacao
        new_base = f"{rotation_part}_{index}_{size_part}"
    else:
        new_base = f"{index}_{size_part}"
    new_file = f"{new_base}{ext}"

    return dst_dir + os.sep + new_file

def get_total_num_of_images_fruits360(split_dir):
    """Conta o total de imagens no Fruits360 para um split (Training/Test)."""
    total = 0
    for cls in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls)
        total += len(os.listdir(cls_path))
    return total

def handle_fruits360_structure(root_dir):
    """Unifica a estrutura do Fruits360 em uma pasta 'raw'."""

    def check_if_already_unified():
        """Verifica se a estrutura ja foi unificada."""
        unified_path = os.path.join(root_dir, "raw")
        return os.path.exists(unified_path)

    def check_if_unified_correctly():
        """Verifica se a estrutura unificada esta correta."""
        unified_path = os.path.join(root_dir, "raw")
        if not os.path.exists(unified_path):
            return False
        for cls in os.listdir(unified_path):
            cls_path = os.path.join(unified_path, cls)
            if not os.path.isdir(cls_path):
                return False
            if len(os.listdir(cls_path)) == 0:
                return False
            # Check a few files (random) to see if they are symlinks
            rand_indexes = np.random.choice(len(os.listdir(cls_path)), size=5, replace=False)
            for idx in rand_indexes:
                fname = os.listdir(cls_path)[idx]
                fpath = os.path.join(cls_path, fname)
                if not os.path.islink(fpath):
                    return False

        return True

    def handle_fruits360_thread_job(cls_path, target_cls_path, split):
        """Job para thread no tratamento do Fruits360."""
        nonlocal i, total_files

        for fname in os.listdir(cls_path):
            src_file = os.path.join(cls_path, fname)
            dst_file = os.path.join(target_cls_path, fname)
            if os.path.exists(dst_file):
                # Arquivo ja existe, trata para evitar conflitos
                if split == "Training": # Se for imagem de treino, entao basta sobrescrever
                    # Remove o arquivo existente
                    os.remove(dst_file)
                else: # Para teste, precisa de fato incrementar o index
                    dst_file = increment_fruits_file_index(dst_file)
                    # Precisa verificar novamente se existe. Se sim, entao o arquivo ja foi copiado
                    if os.path.exists(dst_file):
                        # Remove para sobrescrever
                        os.remove(dst_file)
            # Cria um link simbólico para economizar espaço
            os.symlink(os.path.abspath(src_file), dst_file)
            i += 1
            progress_bar(i, total_files, verbose_text=f"[✓] Processado: {fname} - Salvo em: {dst_file}")

        progress_bar(i, total_files, verbose_text=f"[✓] Concluído o processamento da classe: {cls}")

    if check_if_already_unified():
        if check_if_unified_correctly():
            print(f"Estrutura do Fruits360 já unificada em: {os.path.join(root_dir, 'raw')}")
            return
    
    unified_dir = os.path.join(root_dir, "raw")
    os.makedirs(unified_dir, exist_ok=True)
    threads: list[threading.Thread] = [threading.Thread()] * NUM_THREADS

    for split in ["Training", "Test"]:
        i = 0
        t = 0
        split_path = os.path.join(root_dir, split)
        total_files = get_total_num_of_images_fruits360(split_path)
        progress_bar(i, total_files, verbose_text=f"[...] Processando {split}")

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            target_cls_path = os.path.join(unified_dir, cls)
            os.makedirs(target_cls_path, exist_ok=True)

            while threads[t].is_alive():
                # Espera enquanto nao houver uma thread livre
                t = (t + 1) % NUM_THREADS

            threads[t] = threading.Thread(target=handle_fruits360_thread_job, args=(cls_path, target_cls_path, split))
            threads[t].start()
            t = (t + 1) % NUM_THREADS

        for thread in threads:
            thread.join()

        progress_bar(total_files, total_files, verbose_text=f"[✓] Concluído o processamento do split: {split}", finished=True)
    
    print(f"Estrutura do Fruits360 unificada em: {unified_dir}")

def load_images_from_folder(root_dir):
    """Carrega todas as imagens e rótulos de um dataset estruturado em subpastas."""
    images, labels = [], []
    classes = sorted(os.listdir(root_dir))

    print(f"Classes encontradas: ", end = "")
    if len(classes) <= 10:
        print(", ".join(classes))
    else:
        print("Total de", len(classes), "classes. Listando as primeiras 10: ", end = "")
        print(", ".join(classes[:10]) + ", ...")

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                img_path = os.path.join(cls_path, fname)
                images.append(img_path)
                labels.append(idx)
    return images, labels, classes

def transform(image) -> torch.Tensor:
    return TRANSFORM_COMPOSE(image)

def preprocess_images(image_paths, labels):
    """Aplica transformações e converte para tensor normalizado."""
    # idx_errors: list[int] = []
    # total_images = len(image_paths)
    # progress_bar(0, total_images, verbose_text="[...] Pré-processando imagens")

    # # Cria os arquivos temporarios
    # os.makedirs(TEMP_DIR, exist_ok=True)

    # processed = 0
    # # Processa em batches para controlar uso de memória
    # for batch_start in range(0, total_images, 1000):
    #     batch_image_paths = image_paths[batch_start:batch_start + 1000]
    #     batch_labels = labels[batch_start:batch_start + 1000]

    #     batch_tensors: list[torch.Tensor] = []
    #     batch_label_list: list[int] = []
    #     lock = threading.Lock()

    #     # threads placeholders
    #     threads: list[threading.Thread | None] = [None] * NUM_THREADS

    #     def process(idx, path, lbl, tensors_list, labels_list, lck):
    #         nonlocal idx_errors
    #         try:
    #             with Image.open(path) as img:
    #                 img_rgb = img.convert("RGB")
    #                 tensor = transform(img_rgb)
    #             # Append under lock to keep order of tensors matching labels within this batch
    #             with lck:
    #                 tensors_list.append(tensor.cpu())
    #                 labels_list.append(lbl)
    #         except Exception as e:
    #             print(f"Erro ao processar {path}: {e}")
    #             with lck:
    #                 idx_errors.append(idx)

    #     t = 0
    #     for j, (path, lbl) in enumerate(zip(batch_image_paths, batch_labels)):
    #         global_idx = batch_start + j
    #         # find next available thread slot
    #         while threads[t] is not None and threads[t].is_alive():
    #             t = (t + 1) % NUM_THREADS
    #             time.sleep(0.02)
    #         threads[t] = threading.Thread(target=process, args=(global_idx, path, lbl, batch_tensors, batch_label_list, lock))
    #         threads[t].start()
    #         t = (t + 1) % NUM_THREADS
    #         processed += 1
    #         progress_bar(processed, total_images, verbose_text=f"[✓] Pré-processado: {os.path.basename(path)}")

    #     # Aguarda todas as threads do batch
    #     for thread in threads:
    #         if thread is not None:
    #             thread.join()

    #     # Reporta imagens com erro desse batch
    #     for index in idx_errors:
    #         if batch_start <= index < batch_start + len(batch_image_paths):
    #             print(f"[!] Imagem com erro no pré-processamento: {image_paths[index]}")

    #     # Se houver tensores válidos, salva batch de tensores e labels em arquivos temporários
    #     if len(batch_tensors) > 0:
    #         temp_tensor = torch.stack(batch_tensors)
    #         tmp_tfname = create_new_filename(os.path.join(TEMP_DIR, "temp_tensor.pt"))
    #         torch.save(temp_tensor, tmp_tfname)
    #         del temp_tensor

    #         label_tensor_batch = torch.tensor(batch_label_list, dtype=torch.long)
    #         tmp_lfname = create_new_filename(os.path.join(TEMP_DIR, "temp_labels.pt"))
    #         torch.save(label_tensor_batch, tmp_lfname)
    #         del label_tensor_batch

    #     # libera memoria do batch
    #     del batch_tensors, batch_label_list
    #     gc.collect()

    # Carrega todos os tensores parciais e concatena
    print(f"[...] Carregando tensores parciais e concatenando")
    tensor_files = sorted([os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if "temp_tensor" in f and f.endswith(".pt")])
    label_files = sorted([os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if "temp_labels" in f and f.endswith(".pt")])

    data_tensor = None
    for f, fname in enumerate(tensor_files):
        temp_tensor = torch.load(fname)
        if data_tensor is None:
            data_tensor = temp_tensor
        else:
            data_tensor = torch.cat((data_tensor, temp_tensor), dim=0)
        del temp_tensor
        gc.collect()
        progress_bar(f + 1, len(tensor_files), verbose_text=f"[✓] Carregado e concatenado: {fname}")

    if data_tensor is None:
        # nenhum tensor válido
        data_tensor = torch.empty((0,))
        progress_bar(1, 1, verbose_text=f"[!] Nenhum tensor válido encontrado.", finished=True)
    else:
        progress_bar(len(tensor_files), len(tensor_files), verbose_text=f"[✓] Concatenação concluída.", finished=True)

    # Carrega e concatena labels
    progress_bar(0, len(label_files), verbose_text=f"[...] Carregando rótulos parciais e concatenando")
    label_tensors_list: list[torch.Tensor] = []
    for f, fname in enumerate(label_files):
        temp_labels = torch.load(fname)
        label_tensors_list.append(temp_labels)
        del temp_labels
        gc.collect()
        progress_bar(f + 1, len(label_files), verbose_text=f"[✓] Carregado e armazenado: {fname}")

    if len(label_tensors_list) > 0:
        label_tensor = torch.cat(label_tensors_list, dim=0)
        progress_bar(len(label_files), len(label_files), verbose_text=f"[✓] Concatenação de rótulos concluída.", finished=True)
    else:
        label_tensor = torch.empty((0,), dtype=torch.long)
        progress_bar(1, 1, verbose_text=f"[!] Nenhum rótulo válido encontrado.", finished=True)

    # # Limpa arquivos temporarios
    # for f in os.listdir(TEMP_DIR):
    #     os.remove(os.path.join(TEMP_DIR, f))

    # os.rmdir(TEMP_DIR)

    return data_tensor, label_tensor

def save_split_tensors(X_train, y_train, X_test, y_test, output_dir):
    """Salva tensores em formato .pt para uso no LibTorch."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(X_train, os.path.join(output_dir, "train_images.pt"))
    torch.save(y_train, os.path.join(output_dir, "train_labels.pt"))
    torch.save(X_test, os.path.join(output_dir, "test_images.pt"))
    torch.save(y_test, os.path.join(output_dir, "test_labels.pt"))
    print(f"Tensores salvos em: {output_dir}")


# ==============================
# 🚀 PIPELINE PRINCIPAL
# ==============================

def main():
    print(f"Pré-processando dataset: {DATASET_NAME}")

    # Trata o caso especial do Fruits360, que tem uma estrutura diferente
    # Nele tem uma pasta "Training" e "Test"
    # Antes de tratar, vamos unifiicar tudo em uma pasta "raw"
    if DATASET_NAME == "Fruits360":
        handle_fruits360_structure(os.path.dirname(INPUT_DIR))

    images, labels, classes = load_images_from_folder(INPUT_DIR)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, train_size=TRAIN_RATIO, random_state=SEED, stratify=labels
    )

    # Pré-processar ambos
    X_train_tensor, y_train_tensor = preprocess_images(X_train, y_train)
    X_test_tensor, y_test_tensor = preprocess_images(X_test, y_test)

    # Salvar tensores
    save_split_tensors(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, OUTPUT_DIR)

    # Salvar metadados
    meta = {
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "train_size": len(y_train_tensor),
        "test_size": len(y_test_tensor),
        "seed": SEED
    }
    torch.save(meta, os.path.join(OUTPUT_DIR, "metadata.pt"))
    print("Metadados salvos com sucesso.")


if __name__ == "__main__":
    main()
