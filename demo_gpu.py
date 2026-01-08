import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import tarfile
import sys

def replace_classification_head(model, num_classes):
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        if isinstance(model.classifier, nn.Sequential):
            last_layer_idx = len(model.classifier) - 1
            in_features = model.classifier[last_layer_idx].in_features
            model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)
        else:
            # Cas simple
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

def setup_model(model_name):
    if not hasattr(models, model_name):
        raise ValueError(f"Le modèle {model_name} n'existe pas dans torchvision.models")
    model_creator = getattr(models, model_name)
    return model_creator()

def run(rank, size,model_name="resnet18",dataset="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz" ,bsize=32):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    # --- 1. Set paths ---
    dataset_url = dataset
    download_root = "./"
    filename = dataset_url.split("/")[-1] 
    dataset_folder_name = filename.replace(".tgz", "").replace(".tar.gz", "")
    dataset_folder = os.path.join(download_root, dataset_folder_name)

    # --- 2. Download the dataset ---
    if not os.path.exists(dataset_folder):
        print("Downloading Imagenette...")
        download_url(dataset_url, download_root)
        # Extract
        print("Extracting...")
        with tarfile.open(os.path.join(download_root, filename)) as tar:
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
    model = setup_model(model_name).to(device_id)
    replace_classification_head(model, len(dataset.classes)).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model Resnet18.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    train_images = train_images.to(device_id)
    train_labels = train_labels.to(device_id)
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
    return et - et_read ,et_read - st

if __name__ == "__main__":
    dist.init_process_group("nccl", init_method="env://")
    if len(sys.argv) < 4:
        print("Usage: python demo.py <model_name> <dataset_url> <batch_size> <analyssis_file> <core>")
        sys.exit(1)
    file = sys.argv[4]
    device = int(sys.argv[5])
    systeme_name = sys.argv[1]
    dataset_url = sys.argv[2]
    batch_size = int(sys.argv[3]) 
    size = dist.get_world_size()
    rank = dist.get_rank()
    com,loading = run(rank, size, model_name=systeme_name, dataset=dataset_url, bsize=batch_size)
    if rank == 0:
        with open(file, 'a') as f:
            f.write(f"{loading},{com},{batch_size},{device}\n")
            f.close()