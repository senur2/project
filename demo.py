import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import os
import tarfile
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

def setup_model(model_name):
    if not hasattr(models, model_name):
        raise ValueError(f"Le modèle {model_name} n'existe pas dans torchvision.models")
    model_creator = getattr(models, model_name)
    return model_creator()

def run(rank, size,model_name="resnet18",dataset="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz" ,bsize=32):
    # --- 1. Set paths ---
    dataset_url = dataset
    download_root = "./"
    dataset_folder = os.path.join(download_root, model_name)

    # --- 2. Download the dataset ---
    if not os.path.exists(dataset_folder):
        print("Downloading", model_name,"...")
        download_url(dataset_url, download_root)
        # Extract
        print("Extracting...")
        with tarfile.open(os.path.join(download_root, model_name+"tgz")) as tar:
            tar.extractall(path=download_root)
        print("Done!")

    # --- 3. Define transforms ---
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # --- 4. Load dataset with ImageFolder ---
    dataset = ImageFolder(root=os.path.join(dataset_folder, "train"), transform=transform_train)

    dataset_size = len(dataset)
    localdataset_size = dataset_size//size
    local_dataset = torch.utils.data.Subset(dataset, range(rank*localdataset_size, (rank+1)*localdataset_size))
    sample_size = bsize//size
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)
    model = setup_model (model_name)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model",model,".")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    et_read = time.time()
    print(f'Loading time: {et_read-st} seconds')
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    print(f'Computing + Communication time: {et-et_read} seconds')
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")
    return  et - et_read ,et_read - st

if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    if len(sys.argv) < 4:
        print("Usage: python demo.py <model_name> <dataset_url> <batch_size> <analyssis_file> <core>")
        sys.exit(1)
    file = sys.argv[4]
    device = int(sys.argv[5])
    systeme_name = sys.argv[1],
    dataset_url = sys.argv[2]
    batch_size = int(sys.argv[3]) 
    size = dist.get_world_size()
    rank = dist.get_rank()
    com,loading = run(rank, size, model_name=systeme_name, dataset=dataset_url, bsize=batch_size)
    with open(file, 'w') as f:
        f.write(f"{loading},{com},{device}\n")
        f.close()
   