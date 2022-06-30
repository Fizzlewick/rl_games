from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
import time
import zmq


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class PpoPlayerContinuousZmq(BasePlayer):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        
        #fw-overrides 
        self.is_determenistic = True
        self.env.task.cfg["zmq"] = True

        #cfg["env"]["numEnvs"] = args.num_envs

        self.zmq_context = zmq.Context()
        self.zmq_socket  = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind("tcp://*:5555")
        print("Started ZMQ NetServer on port 5555")

        self.network      = config['network']
        self.actions_num  = self.action_space.shape[0] 
        self.actions_low  = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask         = [False]

        self.normalize_input = self.config['normalize_input']
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()   

    def get_action(self, obs, is_determenistic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())
        return  rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()

    def run(self):
        render           = self.env.task.headless == False
        is_determenistic = True
        has_masks        = False
        has_masks_func   = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        while True:
            # Wait 1 ms for ZMQ message, proccess if avaliable
            if self.zmq_socket.poll(timeout=1) != 0:
                message      = self.zmq_socket.recv()
                zmq_obs_data = np.frombuffer(message, dtype=np.float32)

                zmq_obs_tensor = self.env.task.adapt_zmq_observation(zmq_obs_data)

                #zmq_obs_tensor = torch.from_numpy(zmq_obs_data).to(self.device)
                #zmq_obs_tensor = torch.tensor(zmq_obs_data)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(zmq_obs_tensor, masks, is_determenistic)
                else:
                    action = self.get_action(zmq_obs_tensor, is_determenistic)
                #obses, r, done, info = self.env_step(self.env, action)

                # (-1,1)->rads->filtered rads
                action_adapt = self.env.task.adapt_zmq_actions(action) 

                action_np    = np.array(action_adapt.cpu(), dtype=np.float32)

                self.zmq_socket.send(action_np.tobytes())


                # update sim state if we're rendering
                if render:
                    self.env.task.shadow_zmp_observations(zmq_obs_tensor)
                    self.env.task.gym.simulate(self.env.task.sim)
            else:
                self.env.task.zmq_idle()

            if render:
                self.env.task.render()
