import itertools

from torch.utils.data import Dataset

class DiffEQDataset(Dataset):
    def __init__(self, slice_data):
        self.slice_data = list(itertools.chain.from_iterable(slice_data))
        self.slice_labels = ['dyn' for _ in slice_data[0]]+['reset' for _ in slice_data[1]]+['mix' for _ in slice_data[2]]

    def __len__(self):
        return len(self.slice_labels)

    def __getitem__(self,idx):
        data = self.slice_data[idx]
        label = self.slice_labels[idx]
        return data,label

def diff_eq_data_collate(batch):
    return [(batch_tuple[0],batch_tuple[1]) for batch_tuple in batch]

def grad_to_cpu(cpu_model,cuda_model):
    for cpu_param,cuda_param in zip(cpu_model.parameters(),cuda_model.parameters()):
        if cuda_param.grad is None:
            cpu_param._grad=None
        else:
            cpu_param._grad=cuda_param.grad.cpu()