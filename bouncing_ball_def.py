#!/usr/bin/env python3

import torch
import torch.nn as nn

from torchdiffeq import odeint,odeint_event

class BouncingBall(nn.Module):
    def __init__(self, radius=0.2, gravity=9.8, adjoint=False):
        super().__init__()
        self.register_buffer("initial_pos", torch.tensor(5.0))
        self.register_buffer("initial_vel", torch.tensor(0.0))
        self.register_buffer("gravity", torch.as_tensor(gravity))
        self.register_buffer("radius",torch.as_tensor(radius))
        self.register_buffer("t0",torch.tensor(0.0))
        self.register_buffer("absorption",torch.tensor(0.2))

    def forward(self, t, state):
        pos, vel = state
        dpos = vel
        dvel = -self.gravity
        return dpos, dvel

    def event_fn(self, t, state):
        # positive if ball in mid-air, negative if ball within ground.
        pos, _ = state
        return pos - self.radius

    def state_update(self, state):
        """Updates state based on an event (collision)."""
        pos, vel = state
        pos = (
            pos + 1e-7
        )  # need to add a small eps so as not to trigger the event function immediately.
        vel = -vel * (1 - self.absorption)
        return pos, vel

    def get_collision_times(self, state, nbounces=1):

        event_times = []

        t0 = self.t0

        for i in range(nbounces):
            event_t, solution = odeint_event(
                self,
                state,
                t0,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-8,
                rtol=1e-8,
                odeint_interface=odeint,
            )
            event_times.append(event_t)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return event_times

    def simulate(self, nbounces = 1, initial_state = None, tend = None, tstep = 0.1):
        t0 = self.t0
        device = self.t0.device

        if initial_state:
            state = (initial_state[0], initial_state[1])
        else:
            state = (self.initial_pos, self.initial_vel)
        event_times = self.get_collision_times(state)
        if tend:
            while event_times[-1] < tend:
                nbounces += 1
                event_times = self.get_collision_times(state,nbounces=nbounces)
                if event_times[-1] < event_times[-2] + tstep*10:
                    tend = None
                    event_times = event_times[:-1]
                    nbounces -= 1
                    break

        # get dense path
        trajectory = [torch.atleast_1d(state[0])]
        velocity = [torch.atleast_1d(state[1])]
        times = [torch.atleast_1d(t0)]
        for event_t in event_times:
            #break if less than 3 time steps until next impact
            if event_t - 3*tstep < t0:
                break
            #break if less than 3 time steps until tend
            if tend and tend - 3*tstep < t0:
                break
            #if next event is after tend
            elif tend and event_t >= tend:
                tt = torch.arange(t0, tend-(1e-8), tstep,device=device)
                tt = torch.cat((tt,torch.atleast_1d(tend)))
                solution = odeint(self, state, tt, atol=1e-8, rtol=1e-8)
                trajectory.append(solution[0][1:])
                velocity.append(solution[1][1:])
                times.append(tt[1:])
                t0 = tt[-1]
            else:
                tt = torch.arange(t0,event_t-(1e-8),tstep,device=device)
                last_step_tt_end = tt[-1]+tstep
                tt = torch.cat((tt, torch.atleast_1d(event_t))) #tt now has t0:1/50:event_t, with the last element being exactly event_t
                solution = odeint(self, state, tt, atol=1e-8, rtol=1e-8)
                trajectory.append(solution[0][1:-1])
                velocity.append(solution[1][1:-1])
                times.append(tt[1:-1])

                state = self.state_update(tuple(s[-1] for s in solution))

                last_step_tt = torch.tensor((event_t,last_step_tt_end),device=device)
                last_step_solution = odeint(self, state, last_step_tt, atol=1e-8, rtol=1e-8)
                trajectory.append(last_step_solution[0][1:])
                velocity.append(last_step_solution[1][1:])
                times.append(torch.atleast_1d(last_step_tt_end))

                state = tuple(s[-1] for s in last_step_solution)
                t0 = last_step_tt_end

        if tend and event_times[-1] > tend:
            event_times = event_times[:-1]

        return (
            torch.cat(times),
            torch.cat(trajectory),
            torch.cat(velocity),
            event_times,
        )