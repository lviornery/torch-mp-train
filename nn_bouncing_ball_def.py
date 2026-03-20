#!/usr/bin/env python3\

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_event
#from torchdiffeq import odeint_adjoint as odeint

from mlp_module_def import MLP,FunctionalMLP

class NNBouncingBall(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("initial_pos",torch.tensor([10.0]))
        self.register_buffer("initial_vel",torch.tensor([0.0]))
        self.event_fn = EventFn()
        self.reset_fn = ResetFn()
        self.dynamics_fn = DynamicsFunction(self.event_fn)
        self.event_times_failsafe = 100

    def simulate(self, times, initial_state = None):
        t0 = times[0]

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros([])
            event_fval = self.event_fn(t, state)
            return event_fval

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        if initial_state:
            state = (nn.Parameter(initial_state[0]), nn.Parameter(initial_state[1]), *self.event_fn.event_params)
        else:
            state = (self.initial_pos, self.initial_vel, *self.event_fn.event_params)

        event_times = []

        trajectory = [state[0][None]]
        velocity = [state[1][None]]

        while t0 < times[-1]:
            #get event time
            event_t, solution = odeint_event(
                self.dynamics_fn,
                state,
                t0,
                event_fn=event_fn,
                atol=1e-8,
                rtol=1e-8,
                method="dopri5",
            )

            #interval is the vector t0, all times <= event_t
            interval_ts = times[times > t0]
            if event_t > times[-1] or len(event_times) > self.event_times_failsafe:
                interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])
                #odeint over the interval
                solution_ = odeint(
                    self.dynamics_fn, state, interval_ts, atol=1e-8, rtol=1e-8
                )
                traj_ = solution_[0][1:]
                vel_ = solution_[1][1:]
                trajectory.append(traj_)
                velocity.append(vel_)
                break
            else:
                interval_ts = interval_ts[interval_ts < event_t]
                interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1), event_t.reshape(-1)])
                #skip odeint if we start in an event function
                if interval_ts[-1] > interval_ts[0]:
                    #odeint over the interval
                    solution_ = odeint(
                        self.dynamics_fn, state, interval_ts, atol=1e-8, rtol=1e-8
                    )
                    traj_ = solution_[0][1:-1]  # [0] for position; [1:] to remove intial state.
                    vel_ = solution_[1][1:-1]
                    trajectory.append(traj_)
                    velocity.append(vel_)

                    state = tuple(s[-1] for s in solution)
                else:
                    print("initial event error")

                # update velocity instantaneously.
                state = self.reset_fn(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                pos, *rest = state
                pos = pos + 1e-7 * self.dynamics_fn(event_t, state)[0]
                state = pos, *rest

                event_times.append(event_t)
                t0 = event_t

            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        velocity = torch.cat(velocity, dim=0).reshape(-1)
        return trajectory, velocity, event_times

class EventFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.trigger_net = FunctionalMLP(2,1,32,1)
        self.event_params = nn.ParameterList(self.trigger_net.export_init_params())

    def forward(self, t, state):
        # IMPORTANT: event computation must use variables from the state.
        dyn_state = torch.cat(state[0:2])
        raw_result = self.trigger_net(dyn_state,state[2:])
        return torch.prod(torch.tanh(raw_result))

class ResetFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_net = MLP(2,1,32,1)

    def forward(self, t, state,pre_module=None):
        dyn_state = torch.cat(state[0:2])
        if pre_module is not None:
            vel = self.reset_net(torch.cat(
                (
                    dyn_state,
                    pre_module(dyn_state,state[2:])
                )
            ))
        else:
            vel = self.reset_net(dyn_state)
        return (state[0], vel, *state[2:])

class DynamicsFunction(nn.Module):
    def __init__(self,event_fn):
        super().__init__()
        self.dyn_net = MLP(2,1,32,1)
        for i,p in enumerate(event_fn.event_params):
            self.register_buffer(f"rest_zeros_{i}",torch.zeros_like(p))
        self.extra_state_n = i+1

    def forward(self, t, state):
        dyn_state = torch.cat(state[0:2])
        dvel = self.dyn_net(dyn_state)
        extra_zeros = [getattr(self,f"rest_zeros_{i}") for i in range(self.extra_state_n)]
        return (state[1], dvel, *extra_zeros)