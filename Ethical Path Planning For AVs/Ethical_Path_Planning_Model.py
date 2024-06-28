import argparse
import pickle
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from Ethical_Env import EthicalEnv
# from Testing_Env import TestingEnv



Tauuu = True
parser = argparse.ArgumentParser(description='Solve Ethical Path Planning')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')

parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])
device = torch.device("cpu")
tau = 0.005
# tau = 0.01

class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        # self.fc = nn.Linear(12, 100)
        self.fc = nn.Linear(8, 100)
        self.mu_head = nn.Linear(100, 1)

    def forward(self, s):
        x = F.relu(self.fc(s))
        u = 2.0 * F.tanh(self.mu_head(x))
        return u


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        # self.fc = nn.Linear(13, 100)
        self.fc = nn.Linear(9, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, s, a):
        x = F.relu(self.fc(torch.cat([s, a], dim=1).to(device)))
        state_value = self.v_head(x)
        return state_value


class Memory():
    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():
    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.var = 1.
        self.eval_cnet, self.target_cnet = CriticNet().float().to(device), CriticNet().float().to(device)
        self.eval_anet, self.target_anet = ActorNet().float().to(device), ActorNet().float().to(device)
        # self.memory = Memory(10000)

        # file_to_read = open("data2/stored_object_NEW_BUFF.pickle", "rb")
        file_to_read = open("paramTest1/stored_object_NEW_BUFF.pickle", "rb")
        self.memory = pickle.load(file_to_read)
        file_to_read.close()
        # self.memory = Loaded Memory
        # self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=3e-4)
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)

        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=0.000001)
        # self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)

        # self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-4  )

        # self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-2)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float).to(device))
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return (action.item(),)

    def save_param(self):
        torch.save(self.eval_anet.state_dict(), 'param/ddpg_anet_params.pkl')
        torch.save(self.eval_cnet.state_dict(), 'param/ddpg_cnet_params.pkl')

    def load_paramm(self):
        # Model Nets
        # self.eval_anet.load_state_dict(torch.load('data2/ddpg_anet_params.pkl'))
        # self.eval_cnet.load_state_dict(torch.load('data2/ddpg_cnet_params.pkl'))
        self.eval_anet.load_state_dict(torch.load('param/ddpg_anet_params.pkl', map_location=torch.device('cpu')))
        self.eval_cnet.load_state_dict(torch.load('param/ddpg_cnet_params.pkl', map_location=torch.device('cpu')))


        # Target Nets
        self.target_anet.load_state_dict(self.eval_anet.state_dict())
        self.target_cnet.load_state_dict(self.eval_cnet.state_dict())

    def load_parammACTonly(self):
        # Model Nets
        # self.eval_anet.load_state_dict(torch.load('data2/AstarWeights_actor.pkl'))
        # self.eval_anet.load_state_dict(torch.load('data2/ddpg_anet_params.pkl'))
        # self.eval_cnet.load_state_dict(torch.load('param2/ddpg_cnet_params.pkl'))
        self.eval_anet.load_state_dict(torch.load('paramTest1/ddpg_anet_params.pkl', map_location=torch.device('cpu')))

        # Target Nets
        self.target_anet.load_state_dict(self.eval_anet.state_dict())
        # self.target_cnet.load_state_dict(self.eval_cnet.state_dict())

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self, huhu):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float).to(device)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1).to(device)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1).to(device)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).to(device)

        with torch.no_grad():
            q_target = r + args.gamma * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)

        # update critic net
        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # update actor net
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()
        # if huhu == True:

        # self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        for param, target_param in zip(self.eval_anet.parameters(), self.target_anet.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.eval_cnet.parameters(), self.target_cnet.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # if self.training_step % 100 == 0:
        #     self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        # if self.training_step % 101 == 0:
        #     self.target_anet.load_state_dict(self.eval_anet.state_dict())

        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item()

# set whichenvironenemt you'll be using here
# env = TestingEnv()
env = EthicalEnv()
QArray = []
QRunArray = []

def plot_rewards(training_records, title):
    episodes = [r.ep for r in training_records]
    rewards = [r.reward for r in training_records]
    plt.figure()
    plt.plot(episodes, rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

    episodes = [r.ep for r in training_records]
    rewards = [r.reward for r in training_records]

    plt.figure(figsize=(6, 5))  # Set a larger figure size for better visibility
    plt.plot(episodes, rewards, label='Reward per Episode', linewidth=2, color='blue')  # Increase line width, change color

    plt.title(f'DDPG Training Performance over {len(episodes)} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Moving Averaged Episode Reward')

    # Automatically adjust limits based on the data
    plt.ylim(min(rewards) - 10, max(rewards) + 10)  # Give some padding around the min and max
    plt.xlim(min(episodes) - 1, max(episodes) + 1)

    plt.grid(True)  # Add grid lines for better readability
    plt.legend()  # Add a legend if you have multiple lines

    plt.show()
def main():
    env.seed(args.seed)

    agent = Agent()
    # agent.load_paramm()
    agent.load_parammACTonly()
    # load_trained_agent(agent)

    # plt.ion()
    # ActionsTaken = []
    training_records = []
    training_recordsX = []
    training_recordsX = []
    training_recordsY = []
    running_reward, running_q = -500, 0  # -600,0
    stepscounter = 0
    Tauuuu = True
    Beforee = False

    for i_ep in range(1000):
        score = 0
        state = env.reset()

        for t in range(200):
            stepscounter = stepscounter + 1
            action = agent.select_action(state)
            # done = False
            state_, reward, done, _ = env.step(action)
            print("reward mn el env", reward)
            print("Goal",done)


            score += reward


            agent.store_transition(Transition(state, action, (reward + 8) / 8, state_))
            state = state_

            env.update_plot(state)  # Update the plot with the new state
            plt.pause(0.01)
            # if agent.memory.isfull: ##START TRAINING WHEN MEMORY IS FULLLLLLL
            if 1:  ##START TRAINING WHEN MEMORY IS FULLLLLLL
                q = agent.update(Tauuuu)
                QArray.append(q)
                running_q = 0.99 * running_q + 0.01 * q
                QRunArray.append(running_q)

            if done and t != 199:
                #     # print("loop Braked, final location: ", state_[0:2])
                break;

        # print("reward: ",score)
        # print("actions: ",ActionsTaken)

        running_reward = running_reward * 0.9 + score * 0.1
        print("running_reward",running_reward )
        training_records.append(TrainingRecord(i_ep, running_reward))
        training_recordsY.append(running_reward)
        training_recordsX.append(i_ep)
        if i_ep % args.log_interval == 0:
            print('Step {}\tAverage score: {:.2f}\tAverage Q: {:.2f}\Steps: {}'.format(
                i_ep, running_reward, running_q, stepscounter))

        if (Beforee == False) and running_q > 2:
            agent.optimizer_a.param_groups[0]['lr'] = 3e-4
            tau = 0.005
            Beforee = True
            #     Tauuuu = False
            print("                                          lr = 3e-5 1e-2 and Tau-Deavicated")
        if running_reward > 700:
            print("Solved! Running reward is now {}!".format(running_reward))
            env.close()
            agent.save_param()

            with open('log/ddpg_training_records.pickle', 'wb') as f:
                pickle.dump(training_records, f)
            break

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot

    # buffer_to_save = agent.memory
    # file_to_store = open("stored_object_NEW_BUFF.pickle", "wb")
    # pickle.dump(buffer_to_save, file_to_store)
    # file_to_store.close()
    # print("Buffer Saved")
    # env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    # plt.savefig("img/ddpg.png")
    plt.show()
    plt.ylim(2000, 0)
    plt.xlim(0, 600)



def load_trained_agent(agent):
    try:
        agent.eval_anet.load_state_dict(torch.load('param/ddpg_anet_params.pkl', map_location=torch.device('cpu')))
        agent.target_anet.load_state_dict(agent.eval_anet.state_dict())
        return agent
    except Exception as e:
        print(f"Failed to load the agent: {e}")
        return None

def test_agent(agent, env, num_episodes=3000):
    rewards = []
    running_reward = -400
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(state))  # Convert state to appropriate format if necessary
            next_state, reward, done, _ = env.step(action)
            running_reward = running_reward * 0.9 + reward * 0.1
            print("Reward: ", reward, "Reward: ", running_reward)
            state = next_state
            if running_reward > 600:
                done = True
        rewards.append(running_reward)
    return rewards
def plot_results(training_rewards, test_rewards):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_rewards, label='Training Rewards')
    plt.title('Testing Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_rewards, label='Testing Rewards')
    plt.title('Testing Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

