import os

import matplotlib.pyplot as plt

import torch
import torch.utils.data as tdata
import torch.multiprocessing as mp
import torch.profiler

from bouncing_ball_def import BouncingBall
from nn_bouncing_ball_def import NNBouncingBall
from dataset_utils import DiffEQDataset,diff_eq_data_collate
import training_utils as util
import trajectory_utils as traj_util

#enable CUDA device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "xpu"

torch.set_default_dtype(torch.float64)

#set hyperparameters
NUM_ITERATIONS=1000
MAX_EPOCHS=10
LR_WARMUP = 5
LR_BASE = 0.01
LR_MULT_DICT = {"dyn_dynamics": 1., "event": 1., "reset": 5., "mix_dynamics": 0.1}
EXP_DECAY = 0.99
COS_ALPHA = 0.

DO_PLOT = True
DO_SLICE = True
RAND_TRAJECTORIES = 10
T_STEP = 0.01
BASE_T_END = 3.
BASE_POS = 3.
BASE_VEL = 1.
BASE_ACCEL = 10.
SLICE_STEP=10
TRAJ_LOSS_EXP = 1.2
TRAJ_POWER = 2
THREADS = 6

DIFF_A1 = 0.01
DIFF_A2 = 0.1

lr_obj = util.LearnRateObj(itr=NUM_ITERATIONS, warmup=LR_WARMUP, lr_base=LR_BASE, lr_mult=LR_MULT_DICT, exp_decay=EXP_DECAY, cos_alpha=COS_ALPHA)
traj_loss_obj = traj_util.TrajLossObj(traj_loss_exp=TRAJ_LOSS_EXP,traj_pos_scale=BASE_POS,traj_accel_scale=BASE_ACCEL,traj_power=TRAJ_POWER)

#setup torch
save="figs"
log_folder = "./log/node"
#torch.manual_seed(0) #disable for nondeterministic behavior

def main():
    global lr_obj
    global traj_loss_obj
    with torch.no_grad():
        system = BouncingBall()
        gt_data = traj_util.ground_truth_trajectory(system, RAND_TRAJECTORIES, T_STEP, BASE_T_END, BASE_POS, BASE_VEL)
        slice_data = traj_util.slice_trajectory(gt_data, DO_SLICE, slice_step=SLICE_STEP)
        slice_dataset_list = util.split_data_copies(THREADS,slice_data)
        lr_obj.epoch = MAX_EPOCHS

    model = NNBouncingBall()
    model.train()

    if THREADS == 1:
        slice_dataset = slice_dataset_list[0]
        model.to(device)
        # model parameters - order is event parameters, reset parameters, and dynamics parameters
        param_groups = [{'params': []}, {'params': []}, {'params': []}]
        for parameter in model.event_fn.parameters():
            param_groups[0]['params'].append(parameter)
        for parameter in model.reset_fn.parameters():
            param_groups[1]['params'].append(parameter)
        for parameter in model.dynamics_fn.parameters():
            param_groups[2]['params'].append(parameter)
        single_thread_optimizer = torch.optim.Adam(param_groups)

        single_thread_train_loader = tdata.DataLoader(slice_dataset, shuffle=True, batch_size=1, pin_memory=True, collate_fn=diff_eq_data_collate)
    else:
        mp.set_start_method('spawn', force=True)
        lock = mp.Lock()
        model.share_memory()
        parallel_loader_list = [tdata.DataLoader(slice_dataset, shuffle=True, batch_size=1, collate_fn=diff_eq_data_collate,pin_memory=True, drop_last=True) for slice_dataset in slice_dataset_list]

    #setup plot
    if DO_PLOT:
        fig = plt.figure()
        plt.tight_layout()
        plt.ion()
        os.makedirs(save, exist_ok=True)

    for itr in range(NUM_ITERATIONS):
        if THREADS == 1:
            util.single_train(itr, device, model, single_thread_optimizer, single_thread_train_loader, lr_obj, traj_loss_obj)
        else:
            seed = torch.seed()
            processes = []
            for rank in range(THREADS):
                p = mp.Process(target=util.parallel_train, args=(rank, itr, lock, device, model, parallel_loader_list[rank], lr_obj, traj_loss_obj))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        if DO_PLOT:
            model.eval()
            util.plot_progress(model, gt_data[0][0], fig, itr, save, traj_loss_obj, device=device)
            model.train()
        print("itr "+str(itr))

    if DO_PLOT:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
