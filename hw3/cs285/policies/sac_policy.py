from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def sample(self, obs: torch.FloatTensor):
        action_distribution = self(obs)
        squashed_actions = action_distribution.rsample()

        action_width = (self.action_range[1] - self.action_range[0]) / 2
        action_offset = (self.action_range[1] + self.action_range[0]) / 2
        actions = squashed_actions * action_width + action_offset

        log_probs = action_distribution.log_prob(squashed_actions)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)

        squashed_mean = action_distribution.mean
        mean = squashed_mean * action_width + action_offset

        return actions, log_probs, mean

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action, _, mean = self.sample(observation)

        if not sample:
            return ptu.to_numpy(mean)

        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT:
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file
        loc = self.mean_net(observation)
        batch_size = observation.shape[0]
        logstd_clipped = torch.clamp(self.logstd, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        scale = torch.exp(logstd_clipped)
        scale = scale.repeat(batch_size, 1)

        action_distribution = sac_utils.SquashedNormal(loc, scale)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        observations = ptu.from_numpy(obs)
        actions, log_probs, _ = self.sample(observations)

        q1_t, q2_t = critic(observations, actions)
        q_t = torch.minimum(q1_t, q2_t)

        actor_loss = torch.mean(self.alpha.detach() * log_probs - q_t)
        alpha_loss = torch.mean(-1 * self.alpha * (log_probs.detach() + self.target_entropy))

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha
