from copy import deepcopy

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron = nn.Linear(1,1)

    def forward(self,tensor_input):
        return self.neuron(tensor_input)

def grad_to_cpu(cpu_model,cuda_model):
    for cpu_param,cuda_param in zip(cpu_model.parameters(),cuda_model.parameters()):
        if cuda_param.grad is None:
            cpu_param._grad = None
        else:
            cpu_param._grad=cuda_param.grad.cpu()
            cuda_param.grad = None

def train_model(itr,train_rank,threads,seed,cpu_model,dataset):
    target = torch.tensor([1.], device='cuda')
    with torch.no_grad():
        cuda_model = deepcopy(cpu_model)
        cuda_model.to('cuda:0')
    opt = torch.optim.Adam(cpu_model.parameters())
    parallel_train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=threads, rank=train_rank, seed=seed,
                                                                  drop_last=True)
    parallel_train_loader = torch.utils.data.DataLoader(dataset, sampler=parallel_train_sampler, batch_size=1, pin_memory=True, drop_last=True)

    for epoch in range(len(dataset)//threads):
        parallel_train_sampler.set_epoch(epoch)
        for batch_idx,batch_data in enumerate(parallel_train_loader):
            with torch.no_grad():
                cuda_model.load_state_dict(cpu_model.state_dict())
            opt.zero_grad()
            loss = (cuda_model(batch_data[0].to('cuda',non_blocking=True)) - target)**2
            loss.backward()
            grad_to_cpu(cpu_model,cuda_model)
            opt.step()

if __name__ == "__main__":
    model = SimpleModel()
    dataset = TensorDataset(torch.rand(100))
    threads = 4

    print([p for p in model.parameters()])
    mp.set_start_method("spawn",force=True)
    model.share_memory()
    model.train()

    for itr in range(100):
        processes = []
        seed = torch.seed()
        for rank in range(threads):
            p = mp.Process(target=train_model, args=(itr,rank,threads,seed,model,dataset))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("itr " + str(itr))
        print([p for p in model.parameters()])
