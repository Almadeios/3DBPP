import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DuelingC51(nn.Module):
    def __init__(self, obs_dim, action_dim, atoms, v_min, v_max):
        super(DuelingC51, self).__init__()
        self.atoms = atoms
        self.action_dim = action_dim
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atoms)

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, atoms),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * atoms),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature(x)

        value = self.value_stream(x).view(batch_size, 1, self.atoms)
        advantage = self.advantage_stream(x).view(batch_size, self.action_dim, self.atoms)

        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        probs = F.softmax(q_atoms, dim=2)
        probs = probs.clamp(min=1e-5)  # evitar log(0)
        return probs

    def q_values(self, x):
        probs = self.forward(x)  # [B, A, atoms]
        support = self.support.to(x.device)
        q = torch.sum(probs * support, dim=2)  # [B, A]
        return q


class RainbowAgent:
    def __init__(self, obs_dim, action_dim, args, device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)

        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.lr = args.lr

        self.policy_net = DuelingC51(self.obs_dim, self.action_dim, self.atoms, self.v_min, self.v_max).to(self.device)
        self.target_net = DuelingC51(self.obs_dim, self.action_dim, self.atoms, self.v_min, self.v_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def select_action(self, state, valid_actions=None, evaluate=False):
        if evaluate or random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net.q_values(state)
            if valid_actions is not None:
                # Solo seleccionar entre candidatos válidos
                indices = torch.tensor(valid_actions).long()
                q_valid = q_values[0, indices]
                return valid_actions[q_valid.argmax().item()]
            return q_values.argmax().item()
        else:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)


    def learn(self, batch, indices, weights, memory):
        states, actions, rewards, next_states, dones = batch

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.unsqueeze(1).to(self.device)
        batch_size = states.size(0)

        with torch.no_grad():
            next_dist = self.target_net(next_states)
            next_q = self.policy_net.q_values(next_states).argmax(1)
            next_dist = next_dist[range(batch_size), next_q]

            t_z = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            proj_dist = torch.zeros((batch_size, self.atoms)).to(self.device)
            for i in range(batch_size):
                for j in range(self.atoms):
                    l_idx, u_idx = l[i, j], u[i, j]
                    proj_dist[i, l_idx] += next_dist[i, j] * (u[i, j] - b[i, j])
                    proj_dist[i, u_idx] += next_dist[i, j] * (b[i, j] - l[i, j])

        dist = self.policy_net(states)
        log_p = torch.log(dist[range(batch_size), actions.squeeze(1)])
        losses = -(proj_dist * log_p).sum(1)                 # [batch_size]
        loss = (losses * weights.squeeze()).mean()           # escalar

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for i, idx in enumerate(indices):
            error = losses[i].item()
            memory.update(idx, error)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())