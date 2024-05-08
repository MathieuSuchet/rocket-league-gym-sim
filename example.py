import numpy as np

import rlgym_sim
import time

env = rlgym_sim.make(team_size=3, spawn_opponents=True)

while True:
    obs = env.reset()
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not done:
        actions = np.array([env.action_space.sample() for _ in range(6)])  # agent.act(obs) | Your agent should go here
        new_obs, reward, done, state = env.step(actions)
        ep_reward += sum(reward) / len(reward)
        obs = new_obs
        steps += 1

    length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))