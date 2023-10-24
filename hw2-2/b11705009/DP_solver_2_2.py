import numpy as np
from collections import deque
from gridworld import GridWorld

# from tqdm import tqdm
# import wandb


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
        self.q_values = np.zeros((self.state_space, self.action_space))
        self.policy = np.ones(
            (self.state_space, self.action_space)) / self.action_space
        self.policy_index = np.zeros(self.state_space, dtype=int)
        self.episode_reward = []
        self.episode_loss = []
        self.reward_history = []
        self.loss_history = []

    def record_episode(self, model_name: str) -> None:
        return
        """
        self.reward_history.append(np.average(self.episode_reward))
        self.loss_history.append(np.average(self.episode_loss))
        self.episode_reward = []
        self.episode_loss = []

        wandb.log({f"{model_name}_reward": np.average(
            self.reward_history[-10:]), f"{model_name}_loss": np.average(self.loss_history[-10:])})
        """
    def record_reward(self, reward: float) -> None:
        return 
        #self.episode_reward.append(reward)

    def record_loss(self, loss: float) -> None:
        return 
        #self.episode_loss.append(np.abs(loss))

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index

    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values


class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        assert len(state_trace)-1 == len(action_trace) and len(
            state_trace)-1 == len(reward_trace)
        g_record = []
        T = len(state_trace)-1
        G = np.sum([reward_trace[i]*(self.discount_factor**(i))
                    for i in range(0, T)])
        for t in range(T):

            g_record.append(G)
            self.record_loss(
                G - self.q_values[state_trace[t], action_trace[t]])
            self.q_values[state_trace[t], action_trace[t]] = self.q_values[state_trace[t],
                                                                           action_trace[t]] + self.lr*(G - self.q_values[state_trace[t], action_trace[t]])
            G = (G - reward_trace[t]) / self.discount_factor
        return

        raise NotImplementedError

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        for state in range(self.state_space):
            q = self.q_values[state]
            new_policy = np.full((self.action_space),
                                 self.epsilon / self.action_space)
            new_policy[np.argmax(q)] = self.epsilon / \
                self.action_space + 1 - self.epsilon
            self.policy[state] = new_policy

        return

        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        step_limit = 2000
        state_trace = [current_state]
        action_trace = []
        reward_trace = []
        for iter_episode in range(max_episode):
            step = 0
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            while True:
                # print(self.policy[current_state])
                action = np.random.choice(
                    self.action_space, p=self.policy[current_state])
                next_state, reward, done = self.grid_world.step(action)
                state_trace.append(next_state)
                action_trace.append(action)
                reward_trace.append(reward)
                self.record_reward(reward)
                current_state = next_state
                step += 1
                if done:
                    state_trace[-1] = 666
                    break
            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()
            state_trace = [current_state]
            action_trace = []
            reward_trace = []
            self.record_episode("MC")
            # raise NotImplementedError


class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        # print(s, a, r, s2, a2, is_done)
        if s is None or a is None or r is None:
            return
        elif is_done:
            self.record_loss(r - self.q_values[s, a])
            self.q_values[s, a] = self.q_values[s, a] + \
                self.lr*(r - self.q_values[s, a])
        else:
            self.record_loss(r + self.discount_factor *
                             self.q_values[s2, a2] - self.q_values[s, a])
            self.q_values[s, a] = self.q_values[s, a] + self.lr * \
                (r + self.discount_factor *
                 self.q_values[s2, a2] - self.q_values[s, a])
        q = self.q_values[s]
        # print(q)
        new_policy = np.full((self.action_space),
                             self.epsilon / self.action_space)
        new_policy[np.argmax(q)] = self.epsilon / \
            self.action_space + 1 - self.epsilon
        self.policy[s] = new_policy
        return
        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        for iter_episode in range(max_episode):
            while True:
                # TODO: write your code here
                # hint: self.grid_world.reset() is NOT needed here
                action = np.random.choice(
                    self.action_space, p=self.policy[current_state])
                next_state, reward, done = self.grid_world.step(action)
                self.record_reward(reward)
                self.policy_eval_improve(
                    prev_s, prev_a, prev_r, current_state, action, is_done)
                prev_s = current_state
                prev_a = action
                prev_r = reward
                is_done = done
                current_state = next_state
                if is_done:
                    self.record_episode("SARSA")
                    break

        return
        raise NotImplementedError


class Q_Learning(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.buffer = deque(maxlen=buffer_size)
        self.update_frequency = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))
        return
        raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        return np.random.choice(
            len(self.buffer), size=self.sample_batch_size)

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        if s is None or a is None or r is None:
            return
        elif is_done:
            self.record_loss(r - self.q_values[s, a])
            self.q_values[s, a] = self.q_values[s, a] + \
                self.lr*(r - self.q_values[s, a])
        else:
            self.record_loss((r + self.discount_factor *
                              np.max(self.q_values[s2]) - self.q_values[s, a]))
            self.q_values[s, a] = self.q_values[s, a] + self.lr * \
                (r + self.discount_factor *
                 np.max(self.q_values[s2]) - self.q_values[s, a])
        q = self.q_values[s]
        # print(q)
        new_policy = np.full((self.action_space),
                             self.epsilon / self.action_space)
        new_policy[np.argmax(q)] = self.epsilon / \
            self.action_space + 1 - self.epsilon
        self.policy[s] = new_policy
        return
        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        transition_count = 0
        for iter_episode in range(max_episode):
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            while True:
                # TODO: write your code here
                # hint: self.grid_world.reset() is NOT needed here
                action = np.random.choice(
                    self.action_space, p=self.policy[current_state])
                next_state, reward, done = self.grid_world.step(action)
                self.record_reward(reward)
                self.add_buffer(prev_s, prev_a, prev_r, current_state, is_done)
                transition_count += 1
                if transition_count % self.update_frequency == 0:
                    for iteration_index in self.sample_batch():
                        transition = self.buffer[iteration_index]
                        assert len(transition) == 5
                        self.policy_eval_improve(
                            transition[0], transition[1], transition[2], transition[3], transition[4])
                prev_s = current_state
                prev_a = action
                prev_r = reward
                is_done = done
                current_state = next_state
                if is_done:
                    self.record_episode("Q_Learning")
                    break
        return
        raise NotImplementedError
