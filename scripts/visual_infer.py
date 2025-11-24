import argparse
import json
import os
from typing import List, Protocol, Tuple

import torch
from torch import nn
from PIL import Image
from torchvision.transforms import v2 as T
import tkinter as tk
from tkinter import filedialog


def load_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    classes = j["classes"]  # lista de nomes de classe
    image_h, image_w = j["image_size"]
    norm_mean = j.get("norm_mean", None)
    norm_std = j.get("norm_std", None)
    return classes, (image_h, image_w), norm_mean, norm_std


def build_transform(image_size, norm_mean, norm_std):
    # Replica o pipeline do preprocessamento: Resize + ToImage (uint8),
    # depois converte para float32 [0,1] e normaliza opcionalmente.
    tfms: list[T.Transform] = [
        T.Resize(image_size),
        T.ToImage(),  # uint8 [0,255]
        T.ToDtype(torch.float32, scale=True),  # [0,1]
    ]
    if norm_mean is not None and norm_std is not None and len(norm_mean) == 3 and len(norm_std) == 3:
        tfms.append(T.Normalize(norm_mean, norm_std))
    return T.Compose(tfms)


class SimpleNet(nn.Module):
    """Réplica em Python do modelo C++ definido em Net.hpp (NetImpl).

    Arquitetura:
      Conv2d -> ReLU -> MaxPool
      Conv2d -> ReLU -> MaxPool
      Conv2d -> ReLU -> MaxPool
      (opcional Dropout)
      Flatten -> Linear -> ReLU -> Linear(num_classes)

    Assume entrada redimensionada para 64x64 (como no pré-processamento).
    """

    def __init__(self, num_classes: int, base_filters: int = 32, use_dropout: bool = False):
        super().__init__()
        self.use_dropout = use_dropout

        # Convs espelhando NetImpl (3 convs com padding=1, stride=1, kernel 3x3)
        self.conv1 = nn.Conv2d(3, base_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

        final_spatial = 8  # 64x64 -> /2 ->32 ->/2->16 ->/2->8
        flattened_features = (base_filters * 4) * final_spatial * final_spatial

        self.fc1 = nn.Linear(flattened_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 + ReLU + Pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Conv2 + ReLU + Pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Conv3 + ReLU + Pool
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...  # noqa: D401,E701
    def eval(self) -> "Classifier": ...  # noqa: D401,E701

def load_state_dict_from_json(path):
    with open(path, "r") as f:
        j = json.load(f)

    state_dict = {}
    for name, entry in j.items():
        shape = entry["shape"]
        data = torch.tensor(entry["data"], dtype=torch.float32)
        tensor = data.reshape(shape)
        state_dict[name] = tensor

    return state_dict

def load_model(model_dir_path: str, device: torch.device, num_classes: int) -> Classifier:
    state_dict = load_state_dict_from_json(os.path.join(model_dir_path, "state_dict.json"))
    model = SimpleNet(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def classify_image(
    model: Classifier,
    image_path: str,
    transform: T.Compose,
    classes: List[str],
    device: torch.device,
) -> Tuple[str, float, int, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")
    x = transform(img)  # [C,H,W], float32
    x = x.unsqueeze(0).to(device)  # [1,C,H,W]
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs = torch.softmax(logits, dim=1)[0]
    top_tuple: tuple[torch.Tensor, torch.Tensor] = torch.max(probs, dim=0)
    top_idx: int = int(top_tuple[1].item())
    top_prob: float = top_tuple[0].item()
    pred_class = classes[top_idx]
    return pred_class, top_prob, top_idx, probs.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/Fruits360/test",
        help="Caminho para o diretório do modelo salvo",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/datasets",
        help="Raiz dos datasets (onde fica processed/Fruits360)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Fruits360",
        help="Nome do dataset (para achar metadata.json)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Força uso de CPU mesmo se CUDA estiver disponível",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")

    meta_path = os.path.join(args.data_root, "processed", args.dataset, "metadata.json")
    classes, image_size, norm_mean, norm_std = load_metadata(meta_path)

    transform = build_transform(image_size, norm_mean, norm_std)
    model = load_model(args.model, device, num_classes=len(classes))

    # Loop interativo com diálogo de arquivo
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal

    print("[i] Entrando em modo interativo. Feche o diálogo ou pressione Cancel para sair.")
    while True:
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem para classificar",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Todos os arquivos", "*.*"),
            ],
        )

        if not file_path:
            print("[i] Nenhum arquivo selecionado. Encerrando.")
            break

        pred_class, prob, idx, _ = classify_image(
            model, file_path, transform, classes, device
        )
        print(f"Imagem: {file_path}")
        print(f"Classe prevista: {pred_class} (idx={idx}) com prob={prob:.4f}")


if __name__ == "__main__":
    main()