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
import signal
import threading
import time
import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor, empty, save, from_numpy, ones, manual_seed
from torch import uint8 as uint8_t, long as uint64_t, bool as bool_t
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode

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
manual_seed(SEED)
np.random.seed(SEED)

VERBOSE = True

NUM_THREADS: int = 4

# Proporção do dataset a ser usada (1.0 = 100%).
# Útil para gerar versões menores, rápidas, para testes.
DATASET_USAGE: float = 1.0

# Se True, força o uso de memmap para todos os splits
# (desde que o código de memmap esteja habilitado para aquele split).
FORCE_MEMMAP: bool = False

TRANSFORM_COMPOSE = v2.Compose([
    v2.Resize(IMAGE_SIZE, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
])

RETURN_TOKEN = '\r'
LINE_ABOVE_TOKEN = '\033[F'
LINE_BELOW_TOKEN = '\033[E'
BLANK_LINE_TOKEN = '\n'
END_OF_LINE_TOKEN = '\033[K'

MEGABYTE = 1 << 20
GIGABYTE = 1 << 30

# ==============================
# 🧵 POOL DE THREADS
# ==============================

def worker():
    return # Função placeholder para threads

threads_pool: list[threading.Thread] = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]

# ==============================
# 🧩 FUNÇÕES AUXILIARES
# ==============================

def term_width():
    """Retorna a largura do terminal."""
    return os.get_terminal_size().columns - 1

# Cria handler para tratar signals de interrupcao (Ctrl+C) e limpar a barra
def signal_handler_progress_bar(sig, frame):
    global RETURN_TOKEN, LINE_BELOW_TOKEN, BLANK_LINE_TOKEN

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Restaura comportamento padrão

    for thread in threading.enumerate():
        if thread is not threading.current_thread():
            if thread.is_alive():
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
    
    # Adiciona sufixo indexado em 'size_part'
    if not '-' in size_part:
        size_part = f"{size_part}-1"
    else:
        size_main, size_idx = size_part.split('-')
        size_idx = int(size_idx) + 1
        size_part = f"{size_main}-{size_idx}"

    if len(parts) == 3:
        rotation_part = parts[0]     # Parte da rotacao
        new_base = f"{rotation_part}_{index_part}_{size_part}"
    else:
        new_base = f"{index_part}_{size_part}"
    new_file = f"{new_base}{ext}"

    return dst_dir + os.sep + new_file

def get_total_num_of_images_fruits360(split_dir):
    """Conta o total de imagens no Fruits360 para um split (Training/Test)."""
    total = 0
    for cls in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls)
        total += len(os.listdir(cls_path))
    return total

def check_if_already_unified(total_files: int, unified_dir: str) -> bool:
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
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                count_files += 1
    is_valid = total_files == count_files
    if DATASET_USAGE < 1.0:
        is_valid = count_files >= int(total_files * DATASET_USAGE)
    if not is_valid:
        # Número de arquivos não confere, provavelmente incompleto
        # Remove a pasta unificada para evitar confusão
        import shutil
        shutil.rmtree(unified_dir)
        os.makedirs(unified_dir, exist_ok=True)
        return False
    return True
    

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

    if check_if_already_unified(total_files, unified_dir):
        print(f"Fruits360 structure already unified in: {unified_dir}")
        return

    i = 0
    progress_bar(i, total_files, verbose_text="[...] Processing Fruits360 structure")

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
                    while os.path.exists(dst_file):
                        dst_file = increment_fruits_file_index(dst_file)

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

            while threads_pool[t].is_alive():
                threads_pool[t].join()

            threads_pool[t] = threading.Thread(target=worker, args=(split, cls, cls_path))
            threads_pool[t].start()
            t = (t + 1) % NUM_THREADS

    for thread in threads_pool:
        if thread.is_alive():
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

    def collect_pklot_images(segmented_root: str):
        items = []
        for camera in os.listdir(segmented_root):
            camera_path = os.path.join(segmented_root, camera)
            if not os.path.isdir(camera_path):
                continue
            for climate in os.listdir(camera_path):
                climate_path = os.path.join(camera_path, climate)
                if not os.path.isdir(climate_path):
                    continue
                for date in os.listdir(climate_path):
                    date_path = os.path.join(climate_path, date)
                    if not os.path.isdir(date_path):
                        continue
                    for cls in os.listdir(date_path):
                        cls_path = os.path.join(date_path, cls)
                        if not os.path.isdir(cls_path):
                            continue
                        for fname in os.listdir(cls_path):
                            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                                img_path = os.path.join(cls_path, fname)
                                items.append((cls, img_path))
        return items

    os.makedirs(unified_dir, exist_ok=True)

    # Diretório onde de fato estão as imagens segmentadas
    segmented_root = os.path.join(raw_dir, "PKLotSegmented")
    if not os.path.isdir(segmented_root):
        segmented_root = raw_dir  # fallback: já está em PKLotSegmented

    # Coleta todos os caminhos brutos em paralelo
    all_items = collect_pklot_images(segmented_root)
    total_files = len(all_items)

    if total_files == 0:
        print(f"No images found under PKLot root: {segmented_root}")
        return

    if check_if_already_unified(total_files, unified_dir):
        print(f"PKLot structure already unified in: {unified_dir}")
        return

    i = 0
    progress_bar(i, total_files, verbose_text="[...] Processing PKLot structure")

    def worker(cls_name: str, src_path: str, unified_dir: str):
        nonlocal i
        fname = os.path.basename(src_path)
        target_cls_path = os.path.join(unified_dir, cls_name)
        dst_file = os.path.join(target_cls_path, fname)
        if os.path.exists(dst_file):
            dst_file = create_new_filename(dst_file)
        os.symlink(os.path.abspath(src_path), dst_file)
        i += 1
        progress_bar(i, total_files, verbose_text=f"[✓] Processed: {fname} -> {dst_file}")
        
    t = 0
    for cls_name, src_path in all_items:
        os.makedirs(os.path.join(unified_dir, cls_name), exist_ok=True)
        while threads_pool[t].is_alive():
            threads_pool[t].join()
        threads_pool[t] = threading.Thread(target=worker, args=(cls_name, src_path, unified_dir))
        threads_pool[t].start()
        t = (t + 1) % NUM_THREADS

    for thread in threads_pool:
        if thread.is_alive():
            thread.join()

    progress_bar(total_files, total_files, verbose_text="[✓] Finished PKLot unification", finished=True)
    print(f"PKLot structure unified in: {unified_dir}")


def load_images_from_folder(root_dir):
    """Carrega todas as imagens e rótulos de um dataset estruturado em subpastas."""
    images, labels = [], []
    classes = sorted(os.listdir(root_dir))

    print("Classes encontradas: ", end = "")
    if len(classes) <= 10:
        print(", ".join(classes))
    else:
        print("Total de", len(classes), "classes. Listando as primeiras 10: ", end = "")
        print(", ".join(classes[:10]) + ", ...")

    total_images = 0
    max_idx = -1

    progress_bar(0, len(classes), verbose_text="[...] Carregando imagens e rótulos")
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        total_found = 0
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                img_path = os.path.join(cls_path, fname)
                images.append(img_path)
                labels.append(idx)
                max_idx = max(max_idx, idx)
                total_found += 1
        progress_bar(idx + 1, len(classes), verbose_text=f"Classe '{cls}': {total_found} imagens carregadas")
        total_images += total_found

    progress_bar(len(classes), len(classes), verbose_text=f"[✓] Total de imagens carregadas: {total_images}", finished=True)
    if max_idx + 1 != len(classes):
        print(f"[!] Aviso: rótulos não são contínuos. Máximo idx={max_idx}, mas {len(classes)} classes encontradas.")  
        input("Pressione Enter para continuar...")  

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

def transform(image) -> Tensor:
    return TRANSFORM_COMPOSE(image)

def preprocess_images(image_paths, labels, image_size, use_memmap: bool = False, data_memmap: np.memmap | None=None, label_memmap: np.memmap | None=None, memmap_start: int = 0):
    """Versão reescrita usando ThreadPoolExecutor, context manager em Image.open e coleta periódica de GC.

    Retorna (data_tensor, label_tensor, idx_errors) quando use_memmap==False.
    Quando use_memmap==True retorna (None, None, idx_errors).
    """
    total_images = len(image_paths)
    if total_images == 0:
        if use_memmap:
            return None, None, []
        return (
            empty((0, 3, image_size[0], image_size[1]), dtype=uint8_t),
            empty((0,), dtype=uint64_t),
            [],
        )

    memmap_lock = threading.Lock()
    data_tensor: Tensor | None
    label_tensor: Tensor | None
    idx_errors = []

    if not use_memmap:
        data_tensor = empty((total_images, 3, image_size[0], image_size[1]), dtype=uint8_t)
        label_tensor = empty((total_images,), dtype=uint64_t)
    else:
        data_tensor = None
        label_tensor = None
        if data_memmap is None or label_memmap is None:
            raise ValueError("Memmaps não fornecidas para escrita.")

    progress_bar(0, total_images, verbose_text="[...] Pré-processando imagens")

    def open_image(path: str) -> Tensor:
        img = decode_image(path, mode=ImageReadMode.RGB)
        tensor: Tensor = transform(img)
        return tensor

    def worker_memmap(idx: int, path: str, lbl: int):
        nonlocal data_memmap, label_memmap

        tensor = open_image(path)

        with memmap_lock:
            data_memmap[memmap_start + idx] = tensor.numpy() # type: ignore
            label_memmap[memmap_start + idx] = int(lbl) # type: ignore
        
        del tensor
    
    def worker_tensor(idx: int, path: str, lbl: int):
        nonlocal data_tensor, label_tensor

        tensor = open_image(path)

        data_tensor[idx] = tensor # type: ignore
        label_tensor[idx] = lbl # type: ignore

    processed = 0
    t = 0

    gc.disable()  # Desabilita GC automático para performance
    for idx, (path, lbl) in enumerate(zip(image_paths, labels)):
        while threads_pool[t].is_alive():
            threads_pool[t].join()

        threads_pool[t] = threading.Thread(
            target=worker_memmap if use_memmap else worker_tensor,
            args=(idx, path, lbl)
        )
        threads_pool[t].start()
        t = (t + 1) % NUM_THREADS
        processed += 1
        progress_bar(processed, total_images, verbose_text=f"[✓] Pré-processado: {os.path.basename(path)}")

        if processed % 5000 == 0:
            gc.collect()  # Coleta manual periódica

    for thread in threads_pool:
        if thread.is_alive():
            thread.join()

    gc.enable()  # Reabilita GC automático após processamento
    progress_bar(total_images, total_images, verbose_text="[✓] Pré-processamento concluído", finished=True)

    # Valida as labels invalidas
    num_classes = 0 
    if "PKLot" in image_paths[0]:
        num_classes = 2
    elif "Fruits360" in image_paths[0]:
        num_classes = 225
    if not use_memmap and label_tensor is not None:
        for idx in range(total_images):
            lbl = label_tensor[idx].item()
            if lbl < 0 or lbl >= num_classes:
                idx_errors.append(idx)
        if len(idx_errors) > 0:
            print(f"[!] Aviso: {len(idx_errors)} rótulos inválidos encontrados e serão removidos.")
            input("Pressione Enter para continuar...")
        valid_mask = ones((total_images,), dtype=bool_t)
        for idx in idx_errors:
            valid_mask[idx] = False
        data_tensor = data_tensor[valid_mask]  # type: ignore
        label_tensor = label_tensor[valid_mask]  # type: ignore
    
    return data_tensor, label_tensor


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
        if images is None or labels is None:
            continue
        if split_name not in split_to_prefix:
            continue
        prefix = split_to_prefix[split_name]
        save(images, os.path.join(output_dir, f"{prefix}_images.pt"))
        save(labels, os.path.join(output_dir, f"{prefix}_labels.pt"))

    print(f"Tensores salvos em: {output_dir}")


def estimate_split_bytes(num_images: int) -> int:
    """Estimativa do tamanho em bytes de um split de imagens uint8_t (N, 3, H, W)."""
    return num_images * 3 * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1


def build_tensors_ram(X_split, y_split):
    """Pré-processa um split inteiro em RAM, retornando tensores torch."""
    return preprocess_images(X_split, y_split, IMAGE_SIZE)


def build_tensors_memmap(split_name: str, X_split, y_split, output_dir: str):
    """Cria memmaps em disco para um split específico e retorna tensores torch.

    Gera arquivos `<split>_images.memmap` e `<split>_labels.memmap` em `output_dir`.
    Usa `preprocess_images` com `use_memmap=True` para escrever diretamente.
    """
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, f"{split_name}_images.memmap")
    label_path = os.path.join(output_dir, f"{split_name}_labels.memmap")

    num_images = len(X_split)
    data_mem = np.memmap(
        data_path,
        dtype=np.uint8,
        mode="w+",
        shape=(num_images, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
    )
    label_mem = np.memmap(
        label_path,
        dtype=np.int64,
        mode="w+",
        shape=(num_images,),
    )

    _, _ = preprocess_images(
        X_split,
        y_split,
        IMAGE_SIZE,
        use_memmap=True,
        data_memmap=data_mem,
        label_memmap=label_mem,
        memmap_start=0,
    )

    X_tensor = from_numpy(np.asarray(data_mem)).to(uint8_t)
    y_tensor = from_numpy(np.asarray(label_mem)).long()
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

    # Opcionalmente reduz o tamanho do dataset para testes rápidos
    if not (0.0 < DATASET_USAGE <= 1.0):
        raise ValueError("DATASET_USAGE deve estar no intervalo (0, 1].")

    if DATASET_USAGE < 1.0:
        n_total = len(images)
        n_keep = max(1, int(n_total * DATASET_USAGE))
        images = images[:n_keep]
        labels = labels[:n_keep]
        print(f"[i] DATASET_USAGE={DATASET_USAGE:.3f} -> usando {n_keep} de {n_total} amostras.")

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
        "train": 1 << 32,  # 4 GiB
        "val": 1 << 31,    # 2 GiB
        "test": 1 << 31,   # 2 GiB
    }

    tensors = {}
    memmap_flags = {}

    for split_name, (X_split, y_split) in split_data.items():
        est_bytes = estimate_split_bytes(len(X_split))
        threshold = memmap_thresholds[split_name]
        use_memmap = est_bytes > threshold or FORCE_MEMMAP
        memmap_flags[split_name] = use_memmap

        if use_memmap:
            print(f"[!] Usando memmap para split '{split_name}' (estimado: {est_bytes / (1 << 20):.2f} MiB)")
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
        if name == "PKLot": continue  # Pula PKLot por enquanto
        process_single_dataset(name, in_dir, out_dir, uni_dir)
