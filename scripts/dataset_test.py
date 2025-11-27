import torch, os

data_root = "/home/lucas/Ufpr/torch-cpp-float-benchmark/data/datasets"
processed = os.path.join(data_root, "processed", "Fruits360")

path = os.path.join(processed, "train_images.pt")
print("Loading:", path)
obj = torch.load(path)
print("type:", type(obj))
if isinstance(obj, torch.Tensor):
    print("dtype:", obj.dtype, "shape:", obj.shape)