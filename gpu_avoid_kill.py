import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import os

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(500, 1000)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(1000, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Start running basic DDP example on rank {local_rank}.")
    # create model and move it to GPU with id rank
    device_id = local_rank % torch.cuda.device_count()
    t = torch.eye(128).to('cuda')

    while True:
        # 还有卡间RDMA检测
        t = torch.matmul(t, t.T) 
        t = torch.inverse(t + torch.eye(128).to(device_id) * 1e-4)  
        t = torch.matmul(t, t.T) 
        torch.cuda.empty_cache()
        
    dist.destroy_process_group()

def demo_basic():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(device_id)
    ddp_model = DDP(ToyModel().to(local_rank), device_ids=[local_rank])

    # Create a simple dataset and dataloader
    dataset = TensorDataset(torch.randn(1000, 500), torch.randint(0, 5, (1000,)))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(local_rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Training loop
    while True:  
        for inputs, labels in dataloader:
            # note
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    demo_basic()