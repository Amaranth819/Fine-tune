import gym
import numpy as np
import torch
import torch.nn as nn
import os
import pybulletgym
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from MyReplayBuffer import MyReplayBuffer
from stable_baselines3 import PPO, DDPG, SAC, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


env_name = 'HumanoidPyBulletEnv-v0'
lr = 0.0001
total_timesteps = 5000000
old_log_path = './3.21/33M/'
log_path = './3.21/33M/'
load_old_replay_buffer = False
save_replay_buffer = False
eval_freq = 1
batch_size = 100
replay_buffer_class = MyReplayBuffer
replay_buffer_kwargs = {'counter_interval' : 200000}
tb_log = os.path.join(log_path, 'tb_log')
eval_log_path = os.path.join(log_path, 'eval')
n_cpus = 4


'''
    Expert policy
'''
env = gym.make(env_name)
expert_training_algo = DDPG
expert_model = expert_training_algo.load(os.path.join(old_log_path, str(expert_training_algo.__name__)), env, verbose = 1, learning_rate = lr, tensorboard_log = tb_log, batch_size = batch_size, replay_buffer_class = replay_buffer_class)


'''
    Student policy
'''
# vec_env = make_vec_env(env_name, n_envs = n_cpus, seed = 0)
student_training_algo = SAC
student_model = student_training_algo(
    'MlpPolicy', 
    env, 
    verbose = 1, 
    learning_rate = lr, 
    batch_size = batch_size, 
    replay_buffer_class = replay_buffer_class,
    replay_buffer_kwargs = replay_buffer_kwargs
)


'''
    Generate training dataset
'''
num_interactions = int(1e6)

if isinstance(env.action_space, gym.spaces.Box):
  expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
  expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))

else:
  expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
  expert_actions = np.empty((num_interactions,) + env.action_space.shape)

obs = env.reset()

for i in tqdm(range(num_interactions)):
    action, _ = expert_model.predict(obs, deterministic=True)
    expert_observations[i] = obs
    expert_actions[i] = action
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

np.savez_compressed(
    "expert_data",
    expert_actions=expert_actions,
    expert_observations=expert_observations,
)



from torch.utils.data.dataset import Dataset, random_split
class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


expert_dataset = ExpertDataSet(expert_observations, expert_actions)
train_size = int(0.8 * len(expert_dataset))
test_size = len(expert_dataset) - train_size
train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
)

print("test_expert_dataset: ", len(test_expert_dataset))
print("train_expert_dataset: ", len(train_expert_dataset))


'''
    Pretrain the student policy
'''
def pretrain_agent(
    student,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
      criterion = nn.MSELoss()
    else:
      criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
              # A2C/PPO policy outputs actions, values, log_prob
              # SAC/TD3 policy outputs actions only
              if isinstance(student, (A2C, PPO)):
                action, _, _ = model(data)
              else:
                # SAC/TD3:
                action = model(data)
              action_prediction = action.double()
            else:
              # Retrieve the logits for A2C/PPO when using discrete actions
              dist = model.get_distribution(data)
              action_prediction = dist.distribution.logits
              target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                  # A2C/PPO policy outputs actions, values, log_prob
                  # SAC/TD3 policy outputs actions only
                  if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                  else:
                    # SAC/TD3:
                    action = model(data)
                  action_prediction = action.double()
                else:
                  # Retrieve the logits for A2C/PPO when using discrete actions
                  dist = model.get_distribution(data)
                  action_prediction = dist.distribution.logits
                  target = target.long()

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student_model.policy = model


mean_reward, std_reward = evaluate_policy(student_model, env, n_eval_episodes=10)
print(f"Before cloning: Mean reward = {mean_reward} +/- {std_reward}")

pretrain_agent(
    student_model,
    epochs=10,
    scheduler_gamma=0.7,
    learning_rate=0.001,
    log_interval=100,
    no_cuda=True,
    seed=1,
    batch_size=batch_size,
    test_batch_size=batch_size,
)
student_model_path = './BehaviorClone/SAC_5.17'
student_model.save(student_model_path)


'''
    Evaluate  policies
'''
mean_reward, std_reward = evaluate_policy(expert_model, env, n_eval_episodes=10)
print(f"After cloning: Expert | Mean reward = {mean_reward} +/- {std_reward}")
mean_reward, std_reward = evaluate_policy(student_model, env, n_eval_episodes=10)
print(f"After cloning: Student | Mean reward = {mean_reward} +/- {std_reward}")