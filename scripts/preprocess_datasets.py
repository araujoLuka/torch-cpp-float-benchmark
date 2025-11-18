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

def preprocess_images(image_paths, labels, image_size, use_memmap: bool = False, data_memmap=None, label_memmap=None, memmap_start: int = 0):
    """Aplica transformações e converte para tensor normalizado.

    Se use_memmap for True, escreve diretamente nas memmaps numpy fornecidas:
      - data_memmap: numpy.memmap shape (N, 3, H, W) dtype=np.float32
      - label_memmap: numpy.memmap shape (N,) dtype=np.int64
    Caso contrário, aloca e retorna tensores torch (comportamento anterior).
    """
    idx_errors = []
    idx_errors_lock = threading.Lock()

    def process(idx, path, lbl):
        nonlocal data_tensor, label_tensor, idx_errors
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            if use_memmap:
                # tensor é CPU float32 contíguo; .numpy() irá expor o buffer
                np_arr = tensor.numpy()
                if (data_memmap is None) or (label_memmap is None):
                    raise ValueError("Memmaps não fornecidas para escrita.")
                data_memmap[memmap_start + idx] = np_arr
                label_memmap[memmap_start + idx] = int(lbl)
            else:
                # Copia diretamente para o tensor pré-alocado para evitar muitos objetos temporários
                if data_tensor is None or label_tensor is None:
                    raise ValueError("Tensores não foram alocados corretamente.")
                data_tensor[idx].copy_(tensor)
                label_tensor[idx] = int(lbl)

            img.close()
            del img, tensor
        except Exception as e:
            print(f"Erro ao processar {path}: {e}")
            if use_memmap:
                # Zera slot problemático
                if (data_memmap is None) or (label_memmap is None):
                    raise ValueError("Memmaps não fornecidas para escrita.")
                data_memmap[memmap_start + idx].fill(0)
                label_memmap[memmap_start + idx] = int(lbl)
            else:
                if data_tensor is None or label_tensor is None:
                    raise ValueError("Tensores não foram alocados corretamente.")
                data_tensor[idx].zero_()
                label_tensor[idx] = int(lbl)
            with idx_errors_lock:
                idx_errors.append(idx)

    total_images = len(image_paths)
    i = 0
    t = 0
    progress_bar(i, total_images, verbose_text="[...] Pré-processando imagens")

    if not use_memmap:
        # Aloca buffers grandes para evitar listas de tensores (economiza overhead e referências)
        data_tensor: torch.Tensor | None = torch.empty((total_images, 3, image_size[0], image_size[1]), dtype=torch.float32)
        label_tensor: torch.Tensor | None = torch.empty((total_images,), dtype=torch.long)
    else:
        # placeholders para a clausula "nonlocal" no process
        data_tensor: torch.Tensor | None = None
        label_tensor: torch.Tensor | None = None

    # Inicializa slots de threads
    threads: list[threading.Thread] = []

    for path, lbl in zip(image_paths, labels):
        if len(threads) < NUM_THREADS:
            threads.append(threading.Thread(target=process, args=(i, path, lbl), daemon=True))
        elif len(threads) > NUM_THREADS:
            raise RuntimeError("Número de threads excedeu o limite definido.")
        else:
            while threads[t] is not None and threads[t].is_alive():
                t = (t + 1) % NUM_THREADS
                time.sleep(0.02)  # Evita busy-waiting
            threads[t] = threading.Thread(target=process, args=(i, path, lbl), daemon=True)
        threads[t].start()
        i += 1
        t = (t + 1) % NUM_THREADS
        progress_bar(i, total_images, verbose_text=f"[✓] Pré-processado: {os.path.basename(path)}")
        if (i % 500) == 0:
            gc.collect()  # Coleta de lixo periódica para liberar memória

    for thread in threads:
        if thread is not None:
            thread.join()

    if idx_errors:
        for index in idx_errors:
            print(f"[!] Imagem com erro no pré-processamento: {image_paths[index]}")
        # Filtra imagens com erro para evitar linhas inválidas
        if not use_memmap:
            valid_mask = torch.ones(total_images, dtype=torch.bool)
            for index in idx_errors:
                valid_mask[index] = False
            if valid_mask.sum().item() == 0:
                raise RuntimeError("Todas as imagens falharam no pré-processamento.")
            if data_tensor is None or label_tensor is None:
                raise ValueError("Tensores não foram alocados corretamente.")
            data_tensor = data_tensor[valid_mask]
            label_tensor = label_tensor[valid_mask]
        else:
            # Se usou memmap, opcionalmente poderia recompactar os arquivos em disco. Simplesmente
            # deixamos os slots zerados e retornamos os índices válidos via slicing feito pelo chamador.
            pass

    if use_memmap:
        # Retorna None (os dados ficam nas memmaps passadas) e a máscara de erros para o chamador aplicar
        return None, None, idx_errors
    else:
        return data_tensor, label_tensor, idx_errors


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

    # Se o dataset for grande demais para a RAM, usamos numpy.memmap para escrever em disco em blocos.
    # Estimativa simples do tamanho (bytes) necessário para imagens float32: N * 3 * H * W * 4
    est_train_bytes = len(X_train) * 3 * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 4
    est_test_bytes = len(X_test) * 3 * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 4
    MEMMAP_THRESHOLD = 1 << 30  # 1 GiB, ajuste conforme sua máquina

    use_memmap = (est_train_bytes > MEMMAP_THRESHOLD) or (est_test_bytes > MEMMAP_THRESHOLD)

    if use_memmap:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Cria memmaps para treino e teste
        import numpy as _np
        train_data_path = os.path.join(OUTPUT_DIR, "train_images.memmap")
        train_label_path = os.path.join(OUTPUT_DIR, "train_labels.memmap")
        test_data_path = os.path.join(OUTPUT_DIR, "test_images.memmap")
        test_label_path = os.path.join(OUTPUT_DIR, "test_labels.memmap")

        X_train_mem = _np.memmap(train_data_path, dtype=_np.float32, mode='w+', shape=(len(X_train), 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        y_train_mem = _np.memmap(train_label_path, dtype=_np.int64, mode='w+', shape=(len(X_train),))
        X_test_mem = _np.memmap(test_data_path, dtype=_np.float32, mode='w+', shape=(len(X_test), 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        y_test_mem = _np.memmap(test_label_path, dtype=_np.int64, mode='w+', shape=(len(X_test),))

        # Processa escrevendo direto nas memmaps
        _, _, train_errors = preprocess_images(X_train, y_train, IMAGE_SIZE, use_memmap=True, data_memmap=X_train_mem, label_memmap=y_train_mem, memmap_start=0)
        _, _, test_errors = preprocess_images(X_test, y_test, IMAGE_SIZE, use_memmap=True, data_memmap=X_test_mem, label_memmap=y_test_mem, memmap_start=0)

        # Se houver erros, opcionalmente recompactar (filtrar índices) — aqui criamos tensores a partir das memmaps
        import numpy as _np
        if train_errors:
            mask = _np.ones(len(X_train), dtype=bool)
            mask[train_errors] = False
            X_train_arr = X_train_mem[mask]
            y_train_arr = y_train_mem[mask]
        else:
            X_train_arr = X_train_mem
            y_train_arr = y_train_mem

        if test_errors:
            mask = _np.ones(len(X_test), dtype=bool)
            mask[test_errors] = False
            X_test_arr = X_test_mem[mask]
            y_test_arr = y_test_mem[mask]
        else:
            X_test_arr = X_test_mem
            y_test_arr = y_test_mem

        # Converte para torch sem copiar desnecessariamente (torch.from_numpy compartilha buffer)
        X_train_tensor = torch.from_numpy(_np.asarray(X_train_arr))
        y_train_tensor = torch.from_numpy(_np.asarray(y_train_arr)).long()
        X_test_tensor = torch.from_numpy(_np.asarray(X_test_arr))
        y_test_tensor = torch.from_numpy(_np.asarray(y_test_arr)).long()

    else:
        # Pré-processar ambos em RAM (comportamento anterior)
        X_train_tensor, y_train_tensor, _ = preprocess_images(X_train, y_train, IMAGE_SIZE)
        X_test_tensor, y_test_tensor, _ = preprocess_images(X_test, y_test, IMAGE_SIZE)

    # Salvar tensores
    save_split_tensors(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, OUTPUT_DIR)

    # Salvar metadados
    meta = {
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "train_size": len(y_train_tensor) if y_train_tensor is not None else 0,  
        "test_size": len(y_test_tensor) if y_test_tensor is not None else 0,  
        "seed": SEED,
        "memmap_used": use_memmap
    }
    torch.save(meta, os.path.join(OUTPUT_DIR, "metadata.pt"))
    print("Metadados salvos com sucesso.")


if __name__ == "__main__":
    main()
