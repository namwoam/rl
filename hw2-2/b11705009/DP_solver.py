import numpy as np
import json
from collections import defaultdict

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space = grid_world.get_state_space()
        self.values = np.zeros(self.state_space)

    def get_all_state_values(self) -> np.array:
        return self.values


class MonteCarloPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float):
        """Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        episode_states = []
        episode_rewards = []
        returns = [[] for _ in range(self.grid_world.get_state_space())]
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            episode_states.append(current_state)
            episode_rewards.append(reward)
            if done:
                assert len(episode_states) == len(episode_rewards)
                T = len(episode_states)
                G = 0
                for t in range(T-1, -1, -1):
                    G = self.discount_factor*G + episode_rewards[t]
                    if episode_states[t] not in episode_states[:t]:
                        returns[episode_states[t]].append(G)
                episode_rewards = []
                episode_states = []
            else:
                current_state = next_state
        for state in range(self.grid_world.get_state_space()):
            self.values[state] = np.average(returns[state])


class TDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            if done:
                self.values[current_state] = self.values[current_state] + \
                    self.lr*(reward - self.values[current_state])
            else:
                self.values[current_state] = self.values[current_state] + self.lr*(
                    reward + self.discount_factor*self.values[next_state] - self.values[current_state])

            current_state = next_state


class NstepTDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, num_step: int):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.n = num_step

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        T = 10**8
        t = 0
        episode_states = [current_state]
        episode_rewards = [0]
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            # print(current_state, next_state, reward, done)
            if t < T:
                episode_states.append(next_state)
                episode_rewards.append(reward)
                if done:
                    T = t+1
                    episode_states[-1] = 666
            tau = t - self.n + 1
            if tau >= 0:
                G = np.sum([(self.discount_factor**(i - tau - 1))*episode_rewards[i]
                           for i in range(tau+1, np.min([tau+self.n, T])+1)])
                if tau+self.n < T:
                    G = G+(self.discount_factor**self.n) * \
                        self.values[episode_states[tau+self.n]]
                self.values[episode_states[tau]] = self.values[episode_states[tau]] + \
                    self.lr*(G - self.values[episode_states[tau]])
            current_state = next_state
            if tau == T-1:
                T = 10**8
                t = 0
                episode_states = [current_state]
                episode_rewards = [0]
            else:
                t += 1
