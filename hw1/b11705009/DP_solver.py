from enum import Enum
import numpy as np
import heapq

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
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(
            grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        next_state, reward, done = self.grid_world.step(state, action)
        if done:
            return reward
        q = reward + self.discount_factor * self.values[next_state]
        return q
        # TODO: Get reward from the environment and calculate the q-value
        raise NotImplementedError


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        state_policy = self.policy[state]
        next_state_value = 0
        for index, probability in enumerate(state_policy):
            action = index
            next_state_value += probability * self.get_q_value(state, action)
        # print(f"state value at state-{state} is:{next_state_value}")
        return next_state_value
        raise NotImplementedError

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        delta = np.max(np.abs(self.values - new_values))
        self.values = new_values
        return delta

        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            delta = self.evaluate()
            # print(delta)
            if delta < self.threshold:
                break
        return
        raise NotImplementedError


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        return self.get_q_value(state, self.policy[state])
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            new_values = np.zeros(self.grid_world.get_state_space())
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)
            delta = np.max(np.abs(self.values - new_values))
            self.values = new_values
            # print(f"delta={delta}")
            if delta < self.threshold:
                break
        return
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        new_policy = np.zeros(self.grid_world.get_state_space(), dtype=int)
        for state in range(self.grid_world.get_state_space()):
            best_action = -1
            best_value = float('-inf')
            for action in range(self.grid_world.get_action_space()):
                value = self.get_q_value(state, action)
                if value > best_value:
                    best_action = action
                    best_value = value
            assert best_action != -1
            new_policy[state] = best_action
        return new_policy
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            # print("Evaluating policy")
            self.policy_evaluation()
            # print("Improving policy")
            new_policy = self.policy_improvement()
            if np.all(new_policy == self.policy):
                break
            self.policy = new_policy
        return
        raise NotImplementedError


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        values_from_action = np.zeros(self.grid_world.get_action_space())
        for action in range(self.grid_world.get_action_space()):
            values_from_action[action] = self.get_q_value(state, action)
        return np.max(values_from_action)
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        raise NotImplementedError

    def value_iteration(self):
        new_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        delta = np.max(np.abs(self.values - new_values))
        self.values = new_values
        return delta

    def policy_generation(self):
        for state in range(self.grid_world.get_state_space()):
            best_action = -1
            best_value = float('-inf')
            for action in range(self.grid_world.get_action_space()):
                value = self.get_q_value(state, action)
                if value > best_value:
                    best_action = action
                    best_value = value
            assert best_action != -1
            self.policy[state] = best_action
        return

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            delta = self.value_iteration()
            if delta < self.threshold:
                break
        self.policy_generation()
        return
        raise NotImplementedError


class AsyncDPAlgorithm(Enum):
    INPLACE = 1
    PRIORITIZED = 2
    REALTIME = 3


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        self.Q_table = np.zeros(
            [self.grid_world.get_state_space(), self.grid_world.get_action_space()])

    def run_inplace(self) -> None:
        while True:
            delta = 0
            for state in range(self.grid_world.get_state_space()):
                for action in range(self.grid_world.get_action_space()):
                    self.Q_table[state,
                                 action] = self.get_q_value(state, action)
                new_value = np.max(self.Q_table[state, :])
                delta = np.max([delta, np.abs(new_value - self.values[state])])
                self.values[state] = new_value
            if delta < self.threshold:
                break
        # generate policy with greedy
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax(self.Q_table[state, :])
        return

    def run_inplace_cheat(self) -> None:
        # estimate value
        reward = np.zeros([self.grid_world.get_state_space(),
                          self.grid_world.get_action_space()])
        done = np.zeros([self.grid_world.get_state_space(),
                        self.grid_world.get_action_space()])
        next_state = np.zeros([self.grid_world.get_state_space(
        ), self.grid_world.get_action_space()], dtype=int)
        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                next_state[state, action], reward[state, action], done[state,
                                                                       action] = self.grid_world.step(state, action)

        while True:
            delta = 0
            for state in range(self.grid_world.get_state_space()):
                for action in range(self.grid_world.get_action_space()):
                    self.Q_table[state,
                                 action] = reward[state, action] + self.discount_factor * self.values[next_state[state, action]]
                new_value = np.max(self.Q_table[state, :])
                delta = np.max([delta, np.abs(new_value - self.values[state])])
                self.values[state] = new_value
            if delta < self.threshold:
                break
        # generate policy with greedy
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax(self.Q_table[state, :])
        return

    def run_prioritized(self) -> None:
        internal_policy = np.full(self.policy.shape, -1)
        next_states = np.zeros([self.grid_world.get_state_space(
        ), self.grid_world.get_action_space()], dtype=int)
        prev_states = np.zeros([self.grid_world.get_state_space(
        ), self.grid_world.get_action_space()], dtype=int)
        terminal = []
        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                next_states[state, action], _, done = self.grid_world.step(
                    state, action)
                prev_states[next_states[state, action],
                            action] = state
                if done and state not in terminal:
                    terminal.append(state)
        while True:
            h = []
            for state in range(self.grid_world.get_state_space()):
                heapq.heappush(h, (float("inf"), state))
            delta = 0
            while len(h) != 0:
                heap_content = heapq.heappop(h)
                state = heap_content[1]
                # print("target:", heap_content)
                outcomes = np.zeros(self.grid_world.get_action_space())
                for action in range(self.grid_world.get_action_space()):
                    outcomes[action] = self.get_q_value(state, action)
                best_action = np.random.choice(
                    np.where(outcomes == outcomes.max())[0])
                new_state_value = outcomes[best_action]
                delta = np.max(
                    [np.abs(self.values[state] - new_state_value), delta])
                self.values[state] = new_state_value
                internal_policy[state] = best_action
                for prev_action, prev_state in enumerate(prev_states[state]):
                    if prev_state == state or (prev_action != internal_policy[prev_state] and internal_policy[prev_state] != -1):
                        continue
                    error = np.abs(
                        self.get_q_value(prev_state, prev_action) - self.values[prev_state])
                    if error > 0:
                        heapq.heappush(h, (-error, prev_state))
            # print(delta)
            if delta < self.threshold:
                break
        self.policy = internal_policy
        # generate policy with greedy
        return

    def run_realtime(self) -> None:
        while True:
            delta = 0
            epoch_start_location = self.grid_world.get_state_space()//4
            for start_state in np.random.randint(0, self.grid_world.get_state_space(), size=epoch_start_location):
                current_state = start_state
                while True:
                    for action in range(self.grid_world.get_action_space()):
                        self.Q_table[current_state,
                                     action] = self.get_q_value(current_state, action)
                    best_action = np.random.choice(
                        np.where(self.Q_table[current_state] == self.Q_table[current_state].max())[0])
                    delta = np.max(
                        [delta, np.abs(self.Q_table[current_state, best_action] - self.values[current_state])])
                    self.values[current_state] = self.Q_table[current_state, best_action]
                    next_state, _, done = self.grid_world.step(
                        current_state, best_action)
                    if done:
                        break
                    else:
                        current_state = next_state
            if delta < self.threshold*10:
                break
        # generate policy with greedy
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax(self.Q_table[state, :])
        return

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        np.random.seed(1000)
        self.algorithm_selection = AsyncDPAlgorithm.INPLACE
        if self.algorithm_selection == AsyncDPAlgorithm.INPLACE:
            return self.run_inplace()
        elif self.algorithm_selection == AsyncDPAlgorithm.PRIORITIZED:
            return self.run_prioritized()
        elif self.algorithm_selection == AsyncDPAlgorithm.REALTIME:
            return self.run_realtime()
        raise NotImplementedError
