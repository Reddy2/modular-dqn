import numpy as np
from collections import deque

class NStepRunner:
    def __init__(self, agent, enviornment, n, discount_factor):
        self.agent = agent
        self.env = enviornment
        self._n = n
        self._gammas = np.array([discount_factor**i for i in range(n)])
        self._gamma_n = discount_factor**n
        self._queue = deque(maxlen=n)

    def run_episode(self, render=True):
        # Note: tp1 = t + 1
        t = 0
        state_t = self.env.reset()
        episode_reward = 0
        while True:
            if render:
                self.env.render()

            action_t = self.agent.act(state_t)
            state_tp1, reward_tp1, is_terminal_tp1, _ = self.env.step(action_t)
            self._queue.append((state_t, action_t, reward_tp1))
            state_t = state_tp1
            episode_reward += reward_tp1

            if is_terminal_tp1:
                while self._queue:
                    state_k, action_k = self._queue[0][0], self._queue[0][1]
                    terminal_state = state_tp1
                    rewards = np.array([self._queue[i][2] for i in range(len(self._queue))])
                    reward_kn = np.sum(rewards * self._gammas[:len(self._queue)])
                    self._queue.popleft()
                    
                    # state_tpn is terminal so Q(state_tpn, a) = 0.  For n-step TD we can just set gamma = 0 (instead of Q) to get rid of the term
                    self.agent.observe(state_k, action_k, reward_kn, terminal_state, 0)
                
                return episode_reward

            if len(self._queue) == self._n:
                state_k, action_k = self._queue[0][0], self._queue[0][1]
                state_kpn = state_tp1
                rewards = np.array([self._queue[i][2] for i in range(len(self._queue))])
                reward_kn = np.sum(rewards * self._gammas)
                
                self.agent.observe(state_k, action_k, reward_kn, state_kpn, self._gamma_n)
            
            t += 1
            self.timesteps_ran += 1

    # TODO: Perhaps add in max_timesteps_per_episode parameter
    def run(self, num_episodes=None, num_timesteps=None, render=True):
        if num_episodes == None and num_timesteps == None:
            num_episodes = 1

        self.timesteps_ran = 0
        episodes_ran = 0
        
        while True:
            score = self.run_episode(render)
            episodes_ran += 1

            print("Episode:", episodes_ran, "Score:", score)

            if ((num_episodes is not None and episodes_ran >= num_episodes)
                        or (num_timesteps is not None and self.timesteps_ran >= num_timesteps)):
                break


##class Runner:
##    def __init__(self, agent, enviornment):
##        self.agent = agent
##        self.env = enviornment
##    
##    def run_episode(self, max_timesteps_per_episode=None, render=True):
##        total_reward = 0
##        t = 0
##        state = self.env.reset()
##        
##        while True:
##            if render:
##                self.env.render()
##
##            action = self.agent.act(state)
##            state, reward, is_terminal, _ = self.env.step(action)
##            total_reward += reward
##
##            self.agent.observe(state, reward, is_terminal, self.timesteps_ran)
##            t += 1
##            self.timesteps_ran += 1
##
##            if is_terminal or (max_timesteps_per_episode is not None and t >= max_timesteps_per_episode):
##                return total_reward
##
##    def run(self, num_episodes=None, num_timesteps=None, max_timesteps_per_episode=None, render=True):
##        if num_episodes == None and num_timesteps == None:
##            num_episodes = 1
##
##        self.timesteps_ran = 0
##        episodes_ran = 0
##        
##        while True:
##            score = self.run_episode(max_timesteps_per_episode, render)
##            episodes_ran += 1
##
##            print("Episode:", episodes_ran, "Score:", score)
##
##            if ((num_episodes is not None and episodes_ran >= num_episodes)
##                        or (num_timesteps is not None and self.timesteps_ran >= num_timesteps)):
##                break
