#!/usr/bin/env python3

import math
import numbers
from copy import deepcopy
from typing import List, Dict

import matplotlib.pyplot as plt

import torch
import torch.utils.data as tdata

import trajectory_utils as traj_util
from dataset_utils import DiffEQDataset,diff_eq_data_collate

class LearnRateObj:
    def __init__(self, itr = None, epoch = None, warmup = None, lr_base = None, lr_mult = None, exp_decay = None, cos_alpha=0.):
        self.itr = int(round(itr)) if itr is not None else None
        self.epoch = int(round(epoch)) if epoch is not None else None
        self.warmup = int(round(warmup)) if warmup is not None else None
        self.lr_base = lr_base
        self.lr_mult = lr_mult
        #alpha is the floor above zero of the cosine decay function
        self.exp_decay = exp_decay
        self.cos_alpha = cos_alpha

    def get_lr(self,current_step,exp_decay=False,cos_decay=False):
        new_lr = self.lr_base
        if current_step > self.warmup:
            post_warmup_step = current_step - self.warmup
            if exp_decay:
                new_lr = new_lr*(self.exp_decay**post_warmup_step)
            if cos_decay:
                shifted_cosine = 0.5*(1+math.cos(math.pi*post_warmup_step/self.get_decay_steps()))
                alpha_decay = (1-self.cos_alpha)*shifted_cosine+self.cos_alpha
                new_lr = new_lr*alpha_decay
        else:
            new_lr = (current_step/self.warmup)*self.lr_base
        if isinstance(self.lr_mult, numbers.Number):
            new_lr = self.lr_mult*new_lr
        elif isinstance(self.lr_mult, List):
            new_lr = [mult_lr*new_lr for mult_lr in self.lr_mult]
        elif isinstance(self.lr_mult, Dict):
            new_lr = dict((k,v*new_lr) for k,v in self.lr_mult.items())
        return new_lr

    def get_decay_steps(self):
        return (self.itr * self.epoch) - self.warmup

    def get_epoch_range(self,itr):
        return [epoch_idx+self.epoch*itr for epoch_idx in range(self.epoch)]

def model_params(model):
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append((name,param.data))
    return param_list

def split_data_copies(threads,slice_data):
    dataset_list = []
    partitioned_data = []
    for typed_slice in slice_data:
        partition_slice_data = []
        data_partition_idx = len(typed_slice)//threads
        for thread_idx in range(threads-1):
            partition_slice_data.append(typed_slice[data_partition_idx*thread_idx:data_partition_idx*(thread_idx+1)])
        partition_slice_data.append(typed_slice[data_partition_idx*(thread_idx+1):])
        partitioned_data.append(partition_slice_data)
    for thread_idx in range(threads):
        dataset_list.append(DiffEQDataset([partitioned_data[0][thread_idx],partitioned_data[1][thread_idx],partitioned_data[2][thread_idx]]))
    return dataset_list

def single_train(itr,device,model,optimizer,data_loader,lr_obj,traj_loss_obj):
    for epoch_step in lr_obj.get_epoch_range(itr):
        print("epoch " + str(epoch_step) + " start")
        lr_dict = lr_obj.get_lr(epoch_step,cos_decay=True)
        traj_util.epoch_train(device,model,optimizer,data_loader,lr_dict,traj_loss_obj)
        print("epoch " + str(epoch_step) + " done")

def parallel_train(rank, itr, lock, device, model, data_loader, lr_obj, traj_loss_obj):
    with torch.no_grad():
        cuda_model = deepcopy(model)
        cuda_model.to(device, non_blocking=True)
    #model parameters - order is event parameters, reset parameters, and dynamics parameters
    param_groups = [{'params':[]},{'params':[]},{'params':[]}]
    for parameter in model.event_fn.parameters():
        param_groups[0]['params'].append(parameter)
    for parameter in model.reset_fn.parameters():
        param_groups[1]['params'].append(parameter)
    for parameter in model.dynamics_fn.parameters():
        param_groups[2]['params'].append(parameter)
    parallel_optimizer = torch.optim.Adam(param_groups)

    #parallel_train_sampler = tdata.distributed.DistributedSampler(dataset,num_replicas=threads,rank=rank,seed=seed,drop_last=True)
    #parallel_train_loader = tdata.DataLoader(dataset, sampler=parallel_train_sampler, batch_size=1, collate_fn=diff_eq_data_collate,pin_memory=True, drop_last=True)

    for epoch_step in lr_obj.get_epoch_range(itr):
        print("thread " + str(rank) + " epoch " + str(epoch_step) + " start")
        lr_dict = lr_obj.get_lr(epoch_step,cos_decay=True)
        #parallel_train_sampler.set_epoch(epoch_step)
        traj_util.epoch_train(device, cuda_model, parallel_optimizer, data_loader, lr_dict, traj_loss_obj, cpu_model=model, mp_lock=lock)
        print("thread " + str(rank) + " epoch " + str(epoch_step) + " done")

def plot_progress(model, first_slice_data, fig, itr, save, traj_loss_obj, device="cuda"):
    model_device = next(model.parameters()).device
    if model_device != device:
        model.to(device)
    if first_slice_data.device != device:
        first_slice_data = first_slice_data.to(device)
    times = first_slice_data[0]
    gt_traj = first_slice_data[1]
    initial_pos = gt_traj[:1]
    initial_vel = first_slice_data[2][:1]
    mdl_traj, _, mdl_event_times = model.simulate(times,initial_state=(initial_pos,initial_vel))
    with torch.no_grad():
        loss = traj_util.traj_loss(model, first_slice_data, traj_loss_obj)

    plt.clf()
    plt.plot(
        times.detach().cpu().numpy(),
        gt_traj.detach().cpu().numpy(),
        label="Target",
    )
    plt.plot(
        times.detach().cpu().numpy(),
        mdl_traj.detach().cpu().numpy(),
        label="Learned",
    )
    fig.text(0.5, 0.9, str([itr, loss.detach().cpu().item(), len(mdl_event_times)]), ha='center')
    plt.savefig(f"{save}/{itr:05d}.png")
    plt.draw()
    plt.pause(0.1)
    model.to(model_device)