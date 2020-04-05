
import gym
import argparse
import numpy as np

from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import time
from lib import model, common, experience

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-dir", "--model_dir", required=True, help="Model file to load")
  parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
  args = parser.parse_args()
  env = KukaGymEnv(renders=True, isDiscrete=False, maxSteps=10000000)
  if args.record:
      env = gym.wrappers.Monitor(env, args.record)
  net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
  net.load_state_dict(torch.load(args.model))
  obs = env.reset()
  total_reward = 0.0
  total_steps = 0
  while True:
      obs_v = torch.FloatTensor([obs])
      mu_v = net(obs_v)
      action = mu_v.squeeze(dim=0).data.numpy()
      action = np.clip(action, -1, 1)
      obs, reward, done, _ = env.step(action)
      total_reward += reward
      total_steps += 1
      if done:
          break
  print("In %d steps we got %.3f reward" % (total_steps, total_reward))
if __name__ == "__main__":
  main()
