#!/usr/bin/env python3

import math

import numpy as np

import torch

from dataset_utils import grad_to_cpu

class TrajLossObj:
    def __init__(self, traj_loss_exp=1, traj_pos_scale = 1, traj_accel_scale = 1, traj_power = 1):
        self.traj_loss_exp = traj_loss_exp
        self.traj_pos_scale = traj_pos_scale
        self.traj_accel_scale = traj_accel_scale
        self.traj_power = traj_power
        self.inv_traj_power = 1/traj_power

    def scaled_loss(self,index_time,net_output,ground_truth,quantity):
        time_progress = (index_time-index_time[0])
        time_progress = time_progress/time_progress[-1]
        #power is the power to which loss was taken
        if quantity=='pos':
            inv_loss_scale_factor = self.traj_pos_scale**self.traj_power
        else:
            inv_loss_scale_factor = self.traj_accel_scale**self.traj_power
        loss = torch.pow(net_output-ground_truth,self.traj_power)
        loss = torch.abs(loss)
        loss = torch.div(loss,inv_loss_scale_factor)
        loss = torch.mul(loss,(self.traj_loss_exp**time_progress)/self.traj_loss_exp)
        loss = torch.mean(loss)
        torch.pow(loss,self.inv_traj_power)
        return loss

    def scaled_pos_loss(self,index_time,net_output,ground_truth):
        return self.scaled_loss(index_time,net_output,ground_truth,'pos')

    def scaled_accel_loss(self,index_time,net_output,ground_truth):
        return self.scaled_loss(index_time,net_output,ground_truth,'accel')

def ground_truth_trajectory(system,rand_trajectories,t_step,base_t_end,base_pos,base_vel):
    gt_data = []
    tend = torch.tensor(base_t_end)
    for i in range(rand_trajectories+1):
        if i == 0:
            i_pos = torch.tensor(base_pos)
            i_vel = torch.tensor(base_vel)
        else:
            i_pos = (torch.rand_like(i_pos)*base_pos)+system.radius+base_pos
            i_vel = ((torch.rand_like(i_vel) - 0.5)*base_vel)
        initial_state = (i_pos, i_vel)
        i_obs_times, i_gt_trajectory, i_gt_velocity, i_event_times = system.simulate(tend=tend,
                                                                                     initial_state=initial_state,
                                                                                     tstep=t_step)
        i_gt_acceleration = torch.empty(0)
        for j in range(len(i_obs_times)):
            _,ddx = system.forward(i_obs_times[j],(i_gt_trajectory[j],i_gt_velocity[j]))
            i_gt_acceleration = torch.cat((i_gt_acceleration,torch.atleast_1d(ddx)))
        gt_data.append([
            torch.stack(
                (
                    i_obs_times,
                    i_gt_trajectory,
                    i_gt_velocity,
                    i_gt_acceleration
                )
            ),
            i_event_times
        ])
    return gt_data

def slice_trajectory(gt_data,do_slice,slice_step = 5):
    dyn_slices = []
    reset_slices = []
    mix_slices = []
    for i in range(len(gt_data)):
        i_obs_times = gt_data[i][0][0]
        i_gt_trajectory = gt_data[i][0][1]
        i_gt_velocity = gt_data[i][0][2]
        i_gt_acceleration = gt_data[i][0][3]
        i_event_times = gt_data[i][1]

        #full trajectory as a mixed trajectory no matter what
        mix_time_slice = i_obs_times
        mix_traj_slice = i_gt_trajectory
        mix_vel_slice = i_gt_velocity
        mix_slices.append(torch.stack((mix_time_slice, mix_traj_slice, mix_vel_slice)))
        if do_slice:
            start_index = 0
            event_indices = np.searchsorted(i_obs_times.cpu(), [et.cpu() for et in i_event_times])
            for event_index in event_indices:
                event_index_post = None
                #normal segment pair - dynamics flight followed by reset
                if event_index - slice_step > start_index + math.ceil(slice_step/2) and event_index + slice_step < len(i_obs_times)-math.ceil(slice_step/2):
                    event_index_pre = event_index - slice_step
                    event_index_post = event_index + slice_step
                    dyn_time_slice = i_obs_times[start_index:event_index_pre]
                    reset_time_slice = i_obs_times[event_index_pre:event_index_post]
                    mix_time_slice = i_obs_times[:event_index_post]
                    dyn_traj_slice = i_gt_trajectory[start_index:event_index_pre]
                    reset_traj_slice = i_gt_trajectory[event_index_pre:event_index_post]
                    mix_traj_slice = i_gt_trajectory[:event_index_post]
                    dyn_vel_slice = i_gt_velocity[start_index:event_index_pre]
                    reset_vel_slice = i_gt_velocity[event_index_pre:event_index_post]
                    mix_vel_slice = i_gt_velocity[:event_index_post]
                    dyn_accel_slice = i_gt_acceleration[start_index:event_index_pre]
                    if len(dyn_time_slice) >= slice_step/2:
                        dyn_slices.append(torch.stack((dyn_time_slice,dyn_traj_slice,dyn_vel_slice,dyn_accel_slice)))
                    if len(reset_time_slice) >= slice_step/2:
                        reset_slices.append(torch.stack((reset_time_slice, reset_traj_slice, reset_vel_slice)))
                    if len(mix_time_slice) >= slice_step/2:
                        mix_slices.append(torch.stack((mix_time_slice,mix_traj_slice,mix_vel_slice)))
                #no space for dynamic flight, just a reset
                elif event_index - slice_step <= start_index + math.ceil(slice_step/2) and event_index + slice_step < len(i_obs_times):
                    event_index_post = event_index + slice_step
                    reset_time_slice = i_obs_times[start_index:event_index_post]
                    reset_traj_slice = i_gt_trajectory[start_index:event_index_post]
                    reset_vel_slice = i_gt_velocity[start_index:event_index_post]
                    if len(reset_time_slice) >= slice_step/2:
                        reset_slices.append(torch.stack((reset_time_slice, reset_traj_slice, reset_vel_slice)))
                #no space for post-reset, just dynamic flight
                elif event_index-1 > start_index and event_index + slice_step >= len(i_obs_times)-math.ceil(slice_step/2):
                    dyn_time_slice = i_obs_times[start_index:event_index-1]
                    dyn_traj_slice = i_gt_trajectory[start_index:event_index-1]
                    dyn_vel_slice = i_gt_velocity[start_index:event_index-1]
                    dyn_accel_slice = i_gt_acceleration[start_index:event_index-1]
                    if len(dyn_time_slice) >= slice_step/2:
                        dyn_slices.append(torch.stack((dyn_time_slice,dyn_traj_slice,dyn_vel_slice,dyn_accel_slice)))
                if event_index_post:
                    start_index = event_index_post
    return [dyn_slices, reset_slices, mix_slices]

def epoch_train(device, model, optimizer, data_loader, lr_dict, traj_loss_obj, cpu_model = None, mp_lock = None):
    for batch_idx,batch_data in enumerate(data_loader):
        for batch_slice in batch_data:
            if cpu_model is not None:
                with torch.no_grad():
                    model.load_state_dict(cpu_model.state_dict())
                model.zero_grad()
            optimizer.zero_grad()
            slice_data = batch_slice[0]
            if slice_data.device != device:
                slice_data = slice_data.to(device,non_blocking=True)
            slice_label = batch_slice[1]
            if slice_label == 'dyn':
                optimizer.param_groups[0]['lr'] = 0
                optimizer.param_groups[1]['lr'] = 0
                optimizer.param_groups[2]['lr'] = lr_dict["dyn_dynamics"]
            elif slice_label == 'reset':
                optimizer.param_groups[0]['lr'] = lr_dict["event"]
                optimizer.param_groups[1]['lr'] = lr_dict["reset"]
                optimizer.param_groups[2]['lr'] = 0
            else:
                optimizer.param_groups[0]['lr'] = lr_dict["event"]
                optimizer.param_groups[1]['lr'] = lr_dict["reset"]
                optimizer.param_groups[2]['lr'] = lr_dict["mix_dynamics"]

            loss = traj_loss(model, slice_data, traj_loss_obj)

            loss.backward()
            if cpu_model is not None:
                with torch.no_grad():
                    if mp_lock is not None:
                        mp_lock.acquire()
                        try:
                            grad_to_cpu(cpu_model, model)
                            optimizer.step()
                        finally:
                            mp_lock.release()
                    else:
                        grad_to_cpu(cpu_model, model)
                        optimizer.step()
            else:
                optimizer.step()

def traj_loss(model,slice_data,traj_loss_obj):
    slice_time = slice_data[0]
    slice_traj = slice_data[1]
    slice_vel = slice_data[2]
    if len(slice_data) >= 4:
        slice_accel = slice_data[3]
    else:
        slice_accel = None
    initial_time = slice_time[:1]
    initial_pos = slice_traj[:1]
    initial_vel = slice_vel[:1]

    trajectory, velocity, event_times = model.simulate(slice_time, initial_state=(initial_pos, initial_vel))
    pos_loss = traj_loss_obj.scaled_pos_loss(slice_time,trajectory,slice_traj)
    if slice_accel is not None:
        acceleration = model.dynamics_fn.dyn_net(torch.stack((trajectory, velocity), dim=1))
        accel_loss = traj_loss_obj.scaled_accel_loss(slice_time,acceleration,slice_accel)
    else:
        accel_loss = torch.zeros_like(pos_loss)
    loss = pos_loss + accel_loss
    return loss