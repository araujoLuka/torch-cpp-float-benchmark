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
import numpy as np
import threading
import time
import signal
import torch
from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

# ==============================
# 🔧 CONFIGURAÇÕES DO SCRIPT
# ==============================

DATASET_NAMES = ["Fruits360", "PKLot"]

if any(name not in ["PKLot", "Fruits360"] for name in DATASET_NAMES):
    raise ValueError("DATASET_NAMES deve conter apenas 'PKLot' ou 'Fruits360'.")

DATASET_ROOT = "./data/datasets"

# Trata a posicao do diretorio 'datasets'
# Busca no diretorio atual e no antessessor
# Se nao encontrar, lanca uma excecao
if not os.path.exists(f"{DATASET_ROOT}"):
    if os.path.exists(f"../{DATASET_ROOT}"):
        os.chdir("..")
    else:
        raise FileNotFoundError(f"Diretório '{DATASET_ROOT}' não encontrado.")

INPUT_DIRS = [f"{DATASET_ROOT}/raw/{name}" for name in DATASET_NAMES]
UNIFIED_DIRS = [f"{DATASET_ROOT}/unified/{name}" for name in DATASET_NAMES]
OUTPUT_DIRS = [f"{DATASET_ROOT}/processed/{name}" for name in DATASET_NAMES]

# Tamanho da imagem (redimensionamento padrão)
IMAGE_SIZE = (64, 64)

# Normalização (ImageNet padrão)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD =  [0.229, 0.224, 0.225]

# Proporções globais de divisão do dataset (train/val/test)
GLOBAL_TRAIN_RATIO = 0.7
GLOBAL_VAL_RATIO = 0.1
GLOBAL_TEST_RATIO = 0.2

# Seed para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

VERBOSE = True

NUM_THREADS: int = 4

TRANSFORM_COMPOSE = v2.Compose([
    v2.Resize(IMAGE_SIZE),
    v2.ToImage(),  # Mantém em uint8 [0, 255]
])

RETURN_TOKEN = '\r'
LINE_ABOVE_TOKEN = '\033[F'
LINE_BELOW_TOKEN = '\033[E'
BLANK_LINE_TOKEN = '\n'
END_OF_LINE_TOKEN = '\033[K'

# ==============================
# 🧩 FUNÇÕES AUXILIARES
# ==============================

def term_width():
    """Retorna a largura do terminal."""
    return os.get_terminal_size().columns - 1

# Cria handler para tratar signals de interrupcao (Ctrl+C) e limpar a barra
def signal_handler_progress_bar(sig, frame):
    global RETURN_TOKEN, LINE_BELOW_TOKEN, BLANK_LINE_TOKEN

    # Join nas outras threads
    for thread in threading.enumerate():
        if thread is not threading.current_thread():
            thread.join()

    print(LINE_BELOW_TOKEN)  # Move para a linha abaixo da barra

    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] [!] Processo interrompido pelo usuário.")
    exit(0)


# Apenas imprime o verbose_text se VERBOSE for True
# Verbose text tem que estar antes da barra, ou seja, se for imprimir o texto, entao deve ser apagado toda a barra anterior e feito uma nova
# Sempre tem que ter um espaco em branco antes da barra
# Se VERBOSE e o verbose_text for vazio, nao imprime nada apenas a barra de progresso
# A barra precisa ser atualizada na mesma linha e ter as informacoes de progresso alem de uma informacao visual e de porcentual
# verbose_text sempre deve ser impresso antes da barra
def progress_bar(current, total, verbose_text="", finished=False):
    """Exibe uma barra de progresso no terminal."""
    global VERBOSE, RETURN_TOKEN, LINE_ABOVE_TOKEN, BLANK_LINE_TOKEN

    if not VERBOSE:
        return

    # Ajusta o tamanho do terminal
    BLANK_LINE_TOKEN = ' ' * ((os.get_terminal_size().columns - 1) + 1) + '\n'

    def register_signal_handler():
        if threading.current_thread() is threading.main_thread():
            if not finished:
                # Registra handler customizado
                signal.signal(signal.SIGINT, signal_handler_progress_bar) 
            else:
                # Restaura comportamento padrão
                signal.signal(signal.SIGINT, signal.SIG_DFL)

    fraction = current / total
    percent = fraction * 100
    percent_str = f" ({percent:6.2f}%)"
    progress_str = f"[{current}/{total}] "

    bar_length = term_width() - len(progress_str) - len(percent_str) - 3

    arrow = int(fraction * bar_length) * '#'
    spaces = (bar_length - len(arrow)) * '.'
    
    final_str = ""

    if verbose_text:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        total_str = str(total)
        len_total = len(total_str)
        current_str = str(current).zfill(len_total)
        final_str += f"[{timestamp} - idx:{current_str}] {verbose_text}\n"

    bar_str = ""
    bar_str += RETURN_TOKEN
    bar_str += BLANK_LINE_TOKEN
    bar_str += f"{progress_str}[{arrow}{spaces}]{percent_str}"
    
    if finished:
        bar_str += "\n"  # Nova linha ao finalizar
    else:
        # Deixa a barra pronta para a proxima atualizacao
        bar_str += RETURN_TOKEN
        bar_str += LINE_ABOVE_TOKEN

    register_signal_handler()

    final_str += bar_str

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

def handle_fruits360_structure(raw_dir, unified_dir):
    """Unify Fruits360 structure into a single class-organized directory.

    Original layout (simplified):
        Fruits360/
            Training/
                <Class>/
                    <Image files>
            Test/
                <Class>/
                    <Image files>

    Unified layout:
        unified_fruits360/
            <Class>/
                <symlinked image files from Training and Test>
    """

    def check_if_already_unified(total_files: int) -> bool:
        if not os.path.exists(unified_dir):
            return False
        list_dir = os.listdir(unified_dir)
        if len(list_dir) == 0:
            return False
        count_files = 0
        for cls in list_dir:
            cls_path = os.path.join(unified_dir, cls)
            if not os.path.isdir(cls_path):
                return False
            count_files += len(os.listdir(cls_path))
        if count_files != total_files:
            # Número de arquivos não confere, provavelmente incompleto
            # Remove a pasta unificada para evitar confusão
            import shutil
            shutil.rmtree(unified_dir)
            os.makedirs(unified_dir, exist_ok=True)
            return False
        return True

    os.makedirs(unified_dir, exist_ok=True)

    # Count total images for progress bar
    total_files = 0
    for split in ["Training", "Test"]:
        split_path = os.path.join(raw_dir, split)
        if not os.path.isdir(split_path):
            continue
        total_files += get_total_num_of_images_fruits360(split_path)

    if total_files == 0:
        print(f"No images found under Fruits360 root: {raw_dir}")
        return

    if check_if_already_unified(total_files):
        print(f"Fruits360 structure already unified in: {unified_dir}")
        return

    i = 0
    progress_bar(i, total_files, verbose_text="[...] Processing Fruits360 structure")

    threads: list[threading.Thread] = [threading.Thread()] * NUM_THREADS

    def worker(split: str, cls: str, src_dir: str):
        nonlocal i
        target_cls_path = os.path.join(unified_dir, cls)
        os.makedirs(target_cls_path, exist_ok=True)

        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(target_cls_path, fname)
            if os.path.exists(dst_file):
                if split == "Training":
                    os.remove(dst_file)
                else:
                    dst_file = increment_fruits_file_index(dst_file)
                    if os.path.exists(dst_file):
                        os.remove(dst_file)

            os.symlink(os.path.abspath(src_file), dst_file)
            i += 1
            progress_bar(i, total_files, verbose_text=f"[✓] Processed: {fname} -> {dst_file}")

    t = 0
    for split in ["Training", "Test"]:
        split_path = os.path.join(raw_dir, split)
        if not os.path.isdir(split_path):
            continue

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue

            while threads[t].is_alive():
                t = (t + 1) % NUM_THREADS
                time.sleep(0.001)  # Evita busy-waiting intenso

            threads[t] = threading.Thread(target=worker, args=(split, cls, cls_path))
            threads[t].start()
            t = (t + 1) % NUM_THREADS

    for thread in threads:
        thread.join()

    progress_bar(total_files, total_files, verbose_text="[✓] Finished Fruits360 unification", finished=True)
    print(f"Fruits360 structure unified in: {unified_dir}")

def handle_pklot_structure(raw_dir, unified_dir):
    """Unifica a estrutura do PKLot em uma pasta separada.

    Estrutura original (resumida):
        PKLot/
            PKLotSegmented/
                PUC|UFPR04|UFPR05/
                    <Climate>/
                        <Date>/
                            <Class>/
                                <Image files>

    Estrutura unificada alvo:
        unified_pklot/
            Empty/
                <symlinked image files>
            Occupied/
                <symlinked image files>
    """

    def check_if_already_unified(total_files: int) -> bool:
        if not os.path.exists(unified_dir):
            return False
        list_dir = os.listdir(unified_dir)
        if len(list_dir) == 0:
            return False
        count_files = 0
        for cls in list_dir:
            cls_path = os.path.join(unified_dir, cls)
            if not os.path.isdir(cls_path):
                return False
            count_files += len(os.listdir(cls_path))
        if count_files != total_files:
            # Número de arquivos não confere, provavelmente incompleto
            # Remove a pasta unificada para evitar confusão
            import shutil
            shutil.rmtree(unified_dir)
            os.makedirs(unified_dir, exist_ok=True)
            return False
        return True

    def iter_pklot_images(base_dir: str):
        """Itera sobre todas as imagens e classes na estrutura segmentada do PKLot.

        Otimizada para reduzir chamadas aninhadas de os.listdir e verificações Python.
        """
        if not os.path.isdir(base_dir):
            return

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

        # Ex.: .../PKLotSegmented/PUC/Cloudy/2012-09-12/Empty/img.jpg
        for root, dirs, files in os.walk(base_dir):
            # Classes são exatamente os diretórios 'Empty' e 'Occupied'
            base = os.path.basename(root)
            if base not in ("Empty", "Occupied"):
                continue

            cls = base
            for fname in files:
                if not fname.lower().endswith(valid_exts):
                    continue
                yield cls, os.path.join(root, fname)

    def collect_pklot_images_threaded(base_dir: str) -> list[tuple[str, str]]:
        """Varre PKLotSegmented em paralelo por câmera e retorna (cls, path)."""
        if not os.path.isdir(base_dir):
            return []

        cameras = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        if not cameras:
            return list(iter_pklot_images(base_dir))

        results: list[list[tuple[str, str]]] = [[] for _ in cameras]
        threads_local: list[threading.Thread] = []

        def worker(cam_idx: int, cam_name: str):
            cam_root = os.path.join(base_dir, cam_name)
            for cls, path in iter_pklot_images(cam_root):
                results[cam_idx].append((cls, path))

        for idx, cam in enumerate(cameras):
            t_local = threading.Thread(target=worker, args=(idx, cam), daemon=True)
            threads_local.append(t_local)
            t_local.start()

        for t_local in threads_local:
            t_local.join()

        merged: list[tuple[str, str]] = []
        for chunk in results:
            merged.extend(chunk)
        return merged

    os.makedirs(unified_dir, exist_ok=True)

    # Diretório onde de fato estão as imagens segmentadas
    segmented_root = os.path.join(raw_dir, "PKLotSegmented")
    if not os.path.isdir(segmented_root):
        segmented_root = raw_dir  # fallback: já está em PKLotSegmented

    # Coleta todos os caminhos em paralelo (câmeras em threads)
    all_items = collect_pklot_images_threaded(segmented_root)
    total_files = len(all_items)

    if total_files == 0:
        print(f"No images found under PKLot root: {segmented_root}")
        return

    if check_if_already_unified(total_files):
        print(f"PKLot structure already unified in: {unified_dir}")
        return

    i = 0
    progress_bar(i, total_files, verbose_text="[...] Processing PKLot structure")

    # Threads simples: cada thread processa um lote de caminhos
    threads: list[threading.Thread] = [threading.Thread()] * NUM_THREADS

    def worker(cls_name: str, src_path: str):
        nonlocal i
        target_cls_path = os.path.join(unified_dir, cls_name)
        os.makedirs(target_cls_path, exist_ok=True)

        fname = os.path.basename(src_path)
        dst_file = os.path.join(target_cls_path, fname)
        if os.path.exists(dst_file):
            dst_file = create_new_filename(dst_file)

        os.symlink(os.path.abspath(src_path), dst_file)
        i += 1
        progress_bar(i, total_files, verbose_text=f"[✓] Processed: {fname} -> {dst_file}")

    t = 0
    for cls_name, src_path in all_items:
        while threads[t].is_alive():
            t = (t + 1) % NUM_THREADS
            time.sleep(0.001)  # Evita busy-waiting intenso
        threads[t] = threading.Thread(target=worker, args=(cls_name, src_path))
        threads[t].start()
        t = (t + 1) % NUM_THREADS

    for thread in threads:
        thread.join()

    progress_bar(total_files, total_files, verbose_text="[✓] Finished PKLot unification", finished=True)
    print(f"PKLot structure unified in: {unified_dir}")


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


def split_dataset(images, labels, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Divide o dataset em treino, validação e teste usando porcentagens.

    As proporções devem somar aproximadamente 1.0. A divisão é feita em duas
    etapas para preservar estratificação:
        1) Split train vs (val+test)
        2) Split (val+test) em val vs test
    """
    total = len(images)
    if total != len(labels):
        raise ValueError("Número de imagens e rótulos não coincide.")

    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("Ratios devem estar no intervalo (0, 1).")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("A soma de train_ratio, val_ratio e test_ratio deve ser 1.0.")

    # Primeiro, separa treino do restante (val+test)
    X_train, X_rest, y_train, y_rest = train_test_split(
        images,
        labels,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Dentro do conjunto restante, calcula fração de validação relativa a (val+test)
    remaining_ratio = val_ratio + test_ratio
    if remaining_ratio <= 0.0:
        raise ValueError("Soma de val_ratio e test_ratio deve ser maior que zero.")

    val_ratio_within_rest = val_ratio / remaining_ratio

    X_val, X_test, y_val, y_test = train_test_split(
        X_rest,
        y_rest,
        train_size=val_ratio_within_rest,
        random_state=seed,
        stratify=y_rest,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def transform(image) -> torch.Tensor:
    return TRANSFORM_COMPOSE(image)

def preprocess_images(image_paths, labels, image_size, use_memmap: bool = False, data_memmap=None, label_memmap=None, memmap_start: int = 0):
    """Aplica transformações básicas e converte para tensor.

    Se use_memmap for True, escreve diretamente nas memmaps numpy fornecidas:
        - data_memmap: numpy.memmap shape (N, 3, H, W) dtype=np.uint8
        - label_memmap: numpy.memmap shape (N,) dtype=np.int64
    Caso contrário, aloca e retorna tensores torch (comportamento anterior).
    """
    total_images = len(image_paths)
    if total_images == 0:
        if use_memmap:
            return None, None, []
        return (
            torch.empty((0, 3, image_size[0], image_size[1]), dtype=torch.uint8),
            torch.empty((0,), dtype=torch.long),
            [],
        )

    idx_errors: list[int] = []
    idx_errors_lock = threading.Lock()

    if not use_memmap:
        data_tensor: torch.Tensor | None = torch.empty(
            (total_images, 3, image_size[0], image_size[1]),
            dtype=torch.uint8,
        )
        label_tensor: torch.Tensor | None = torch.empty(
            (total_images,),
            dtype=torch.long,
        )
    else:
        data_tensor = None
        label_tensor = None

    def worker(idx: int, path: str, lbl: int):
        nonlocal data_tensor, label_tensor, idx_errors
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            if use_memmap:
                if (data_memmap is None) or (label_memmap is None):
                    raise ValueError("Memmaps não fornecidas para escrita.")
                # tensor está em uint8; garante dtype ao gravar
                np_arr = tensor.numpy().astype(np.uint8)
                data_memmap[memmap_start + idx] = np_arr
                label_memmap[memmap_start + idx] = int(lbl)
            else:
                if data_tensor is None or label_tensor is None:
                    raise ValueError("Tensores não foram alocados corretamente.")
                data_tensor[idx].copy_(tensor)
                label_tensor[idx] = int(lbl)

            img.close()
            del img, tensor
        except Exception as e:
            print(f"Erro ao processar {path}: {e}")
            if use_memmap:
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

    progress_bar(0, total_images, verbose_text="[...] Pré-processando imagens")

    num_workers = min(NUM_THREADS, total_images)
    threads: list[threading.Thread | None] = [None] * num_workers
    next_thread = 0
    processed = 0

    gc.disable()  # Desabilita coleta automática para performance
    for idx, (path, lbl) in enumerate(zip(image_paths, labels)):
        while threads[next_thread] is not None and threads[next_thread].is_alive():
            next_thread = (next_thread + 1) % num_workers
            time.sleep(0.001)  # Evita busy-waiting intenso

        threads[next_thread] = threading.Thread(
            target=worker,
            args=(idx, path, int(lbl)),
            daemon=True,
        )
        threads[next_thread].start()

        processed += 1
        # Atualiza barra apenas a cada 1 imagem (ou na última) para reduzir overhead de I/O
        if (processed % 1) == 0 or processed == total_images:
            progress_bar(
                processed,
                total_images,
                verbose_text=f"[✓] Pré-processado: {os.path.basename(path)}",
            )
        if (processed % 5000) == 0:
            gc.collect()

        next_thread = (next_thread + 1) % num_workers

    gc.enable()  # Reabilita coleta automática após processamento

    for t in threads:
        if t is not None:
            t.join()
    
    progress_bar(total_images, total_images, verbose_text="[✓] Pré-processamento concluído", finished=True)

    if idx_errors:
        for index in idx_errors:
            print(f"[!] Imagem com erro no pré-processamento: {image_paths[index]}")
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
            # Se usou memmap, mantemos slots zerados; o chamador pode recompactar depois.
            pass

    if use_memmap:
        return None, None, idx_errors
    return data_tensor, label_tensor, idx_errors


def save_split_tensors(splits, output_dir):
    """Salva tensores de treino, validação e teste em formato .pt.

    Espera um dicionário no formato:
        {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    split_to_prefix = {
        "train": "train",
        "val": "val",
        "test": "test",
    }

    for split_name, (images, labels) in splits.items():
        if split_name not in split_to_prefix:
            continue
        prefix = split_to_prefix[split_name]
        torch.save(images, os.path.join(output_dir, f"{prefix}_images.pt"))
        torch.save(labels, os.path.join(output_dir, f"{prefix}_labels.pt"))

    print(f"Tensores salvos em: {output_dir}")


def estimate_split_bytes(num_images: int) -> int:
    """Estimativa do tamanho em bytes de um split de imagens uint8 (N, 3, H, W)."""
    return num_images * 3 * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1


def build_tensors_ram(X_split, y_split):
    """Pré-processa um split inteiro em RAM, retornando tensores torch."""
    X_tensor, y_tensor, _ = preprocess_images(X_split, y_split, IMAGE_SIZE)
    return X_tensor, y_tensor


def build_tensors_memmap(split_name: str, X_split, y_split, output_dir: str):
    """Cria memmaps em disco para um split específico e retorna tensores torch.

    Gera arquivos `<split>_images.memmap` e `<split>_labels.memmap` em `output_dir`.
    Usa `preprocess_images` com `use_memmap=True` para escrever diretamente.
    """
    import numpy as _np

    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, f"{split_name}_images.memmap")
    label_path = os.path.join(output_dir, f"{split_name}_labels.memmap")

    num_images = len(X_split)
    data_mem = _np.memmap(
        data_path,
        dtype=_np.uint8,
        mode="w+",
        shape=(num_images, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
    )
    label_mem = _np.memmap(
        label_path,
        dtype=_np.int64,
        mode="w+",
        shape=(num_images,),
    )

    _, _, idx_errors = preprocess_images(
        X_split,
        y_split,
        IMAGE_SIZE,
        use_memmap=True,
        data_memmap=data_mem,
        label_memmap=label_mem,
        memmap_start=0,
    )

    if idx_errors:
        mask = _np.ones(num_images, dtype=bool)
        mask[idx_errors] = False
        data_arr = data_mem[mask]
        label_arr = label_mem[mask]
    else:
        data_arr = data_mem
        label_arr = label_mem

    X_tensor = torch.from_numpy(_np.asarray(data_arr)).to(torch.uint8)
    y_tensor = torch.from_numpy(_np.asarray(label_arr)).long()
    return X_tensor, y_tensor

def process_single_dataset(dataset_name, input_dir, output_dir, unified_dir):
    """Processa um único dataset."""
    print(f"Pré-processando dataset: {dataset_name}")

    # Trata casos especiais de estrutura dos datasets (ex.: Fruits360 com "Training"/"Test",
    # e PKLot com subpastas por câmera/clima). Unifica/normaliza os diretórios brutos
    # em uma estrutura 'unified' pronta para o processamento.
    if dataset_name == "Fruits360":
        handle_fruits360_structure(input_dir, unified_dir)
    elif dataset_name == "PKLot":
        handle_pklot_structure(input_dir, unified_dir)
    else:
        raise ValueError(f"Dataset desconhecido: {dataset_name}")

    # Após unificar, sempre carregamos a partir de unified_dir
    images, labels, classes = load_images_from_folder(unified_dir)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        images,
        labels,
        train_ratio=GLOBAL_TRAIN_RATIO,
        val_ratio=GLOBAL_VAL_RATIO,
        test_ratio=GLOBAL_TEST_RATIO,
        seed=SEED,
    )

    # Estima bytes por split e decide uso de memmap de forma diferenciada
    split_data = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    memmap_thresholds = {
        "train": 1 << 33,  # 8 GiB
        "val": 1 << 33,    # 8 GiB
        "test": 1 << 33,   # 8 GiB
    }

    tensors = {}
    memmap_flags = {}

    for split_name, (X_split, y_split) in split_data.items():
        est_bytes = estimate_split_bytes(len(X_split))
        threshold = memmap_thresholds[split_name]
        use_memmap = est_bytes > threshold
        memmap_flags[split_name] = use_memmap

        if use_memmap:
            X_tensor, y_tensor = build_tensors_memmap(split_name, X_split, y_split, output_dir)
        else:
            X_tensor, y_tensor = build_tensors_ram(X_split, y_split)

        tensors[split_name] = (X_tensor, y_tensor)

    (X_train_tensor, y_train_tensor) = tensors["train"]
    (X_val_tensor, y_val_tensor) = tensors["val"]
    (X_test_tensor, y_test_tensor) = tensors["test"]

    # Salvar tensores (train/val/test)
    save_split_tensors(
        {
            "train": (X_train_tensor, y_train_tensor),
            "val": (X_val_tensor, y_val_tensor),
            "test": (X_test_tensor, y_test_tensor),
        },
        output_dir,
    )

    # Salvar metadados em JSON, fácil de ler em C++
    import json

    meta = {
        "classes": classes,
        "image_size": list(IMAGE_SIZE),
        "train_size": int(len(y_train_tensor) if y_train_tensor is not None else 0),
        "val_size": int(len(y_val_tensor) if y_val_tensor is not None else 0),
        "test_size": int(len(y_test_tensor) if y_test_tensor is not None else 0),
        "seed": int(SEED),
        "memmap_used": {k: bool(v) for k, v in memmap_flags.items()},
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "dtype_images": "uint8",
        "dtype_labels": "int64",
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Metadados salvos em: {meta_path}")

if __name__ == "__main__":
    # Executa o pipeline para todos os datasets configurados
    for name, in_dir, uni_dir, out_dir in zip(DATASET_NAMES, INPUT_DIRS, UNIFIED_DIRS, OUTPUT_DIRS):
        process_single_dataset(name, in_dir, out_dir, uni_dir)
