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

def run(rank, size,model="resnet18",dataset="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz" ,bsize=32):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    # --- 1. Set paths ---
    dataset_url = dataset
    download_root = "./"
    dataset_folder = os.path.join(download_root, model)

    # --- 2. Download the dataset ---
    if not os.path.exists(dataset_folder):
        print("Downloading Imagenette...")
        download_url(dataset_url, download_root)
        # Extract
        print("Extracting...")
        with tarfile.open(os.path.join(download_root, model + "tgz")) as tar:
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
    model = models.resnet18().to(device_id)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device_id)
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
    dist.init_process_group("gloo", init_method="env://")
    if len(sys.argv) < 4:
        print("Usage: python demo.py <model_name> <dataset_url> <batch_size> <analyssis_file>")
        print("Using default parameters.")
    file = sys.argv[4]
    systeme_name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    dataset_url = sys.argv[2] if len(sys.argv) > 2 else "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    size = dist.get_world_size()
    rank = dist.get_rank()
    com,loading = run(rank, size, model=systeme_name, dataset=dataset_url, batch_size=batch_size)
    with open(file, 'w') as f:
        f.write(f"{loading},{com}, {batch_size}\n")
        f.close()