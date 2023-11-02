import json
import os
import warnings
from time import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from PIL import Image

from gridworld import GridWorld

warnings.filterwarnings("ignore")

STEP_REWARD = -0.01
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
EXIT_REWARD = 0.1
BAIT_REWARD = 1.0
BAIT_STEP_PENALTY = -0.25
MAX_STEP = 1000
RENDER_MODE = "ansi"


def print_sa(obs, action, env):
    print("state:", obs[0], "action: ", env.grid_world.ACTION_INDEX_TO_STR[action[0]])


def test_task(filename="tasks/lava.txt", algorithm = PPO):
    task_name = os.path.split(filename)[1].replace(".txt", "")
    print("Task: ", task_name)
    gym.register(f"GridWorld{task_name.capitalize()}-v1", entry_point="gridworld:GridWorldEnv")
    env = gym.make(
        f"GridWorld{task_name.capitalize()}-v1",
        render_mode=RENDER_MODE,
        maze_file=filename,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        exit_reward=EXIT_REWARD,
        bait_reward=BAIT_REWARD,
        bait_step_penalty=BAIT_STEP_PENALTY,
        max_step=MAX_STEP,
    )
    env.grid_world.visualize(f"{task_name}.png")
    model = algorithm.load(f"assets/gridworld_{task_name}", env=env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    # random render 3 trajectories
    for i in range(3):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            vec_env.render()
            # you can use pring_sa to debug state action pair
            # print_sa(obs, action, env)
            obs, reward, done, info = vec_env.step(action)


def test_speed(filename="tasks/maze.txt"):
    task_name = os.path.split(filename)[1].replace(".txt", "")
    grid_world = GridWorld(
        maze_file=filename,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        exit_reward=EXIT_REWARD,
        bait_reward=BAIT_REWARD,
        bait_step_penalty=BAIT_STEP_PENALTY,
        max_step=MAX_STEP,
    )
    t = time()
    task_id = 0
    with open(os.path.join("grading", f"test_case_{task_name}.json"), "r") as f:
        all_task = json.load(f)
    grid_world.set_current_state(all_task[task_id])
    for _ in range(100000):
        action = np.random.choice(grid_world.get_action_space())
        obs, reward, done, truncated = grid_world.step(action)
        if done:
            task_id += 1
            grid_world.reset()
            grid_world.set_current_state(all_task[task_id])
    print(f"time for task {task_name}: {np.round(time() - t, 4)}")


def test_correctness(filename="tasks/maze.txt"):
    task_name = os.path.split(filename)[1].replace(".txt", "")
    grid_world = GridWorld(
        maze_file=filename,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        exit_reward=EXIT_REWARD,
        bait_reward=BAIT_REWARD,
        bait_step_penalty=BAIT_STEP_PENALTY,
        max_step=MAX_STEP,
    )
    df = pd.read_csv(os.path.join("grading", f"gridworld_{task_name}.csv"))
    state = df["state"]
    action = df["action"]
    reward = df["reward"]
    done = df["done"]
    truncated = df["truncated"]
    next_state = df["next_state"]
    result = []
    grid_world.set_current_state(state[0])
    for _, a, r, d, t, ns in zip(state, action, reward, done, truncated, next_state):
        next_state_prediction, reward_prediction, done_prediction, truncated_prediction = grid_world.step(a)
        if done_prediction:
            next_state_prediction = grid_world.reset()
            result.append( (next_state_prediction in grid_world._init_states) and reward_prediction == r and done_prediction == d and truncated_prediction == t)
            grid_world.set_current_state(ns)
        else:
            result.append(next_state_prediction == ns and reward_prediction == r and done_prediction == d and truncated_prediction == t)

    print(f"The correctness of the task {task_name}: {np.round(np.mean(result) * 100, 2)} %")

def write_gif(filename="lava.txt", algorithm = PPO):
    task_name = os.path.split(filename)[1].replace(".txt", "")
    gym.register(f"GridWorld{task_name.capitalize()}-v1", entry_point="gridworld:GridWorldEnv")
    env = gym.make(
        f"GridWorld{task_name.capitalize()}-v1",
        render_mode=RENDER_MODE,
        maze_file=filename,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        exit_reward=EXIT_REWARD,
        bait_reward=BAIT_REWARD,
        bait_step_penalty=BAIT_STEP_PENALTY,
        max_step=MAX_STEP,
    )
    env.grid_world.visualize(f"{task_name}.png")
    model = algorithm.load(f"assets/gridworld_{task_name}", env=env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    states = []
    while True:
        action, _states = model.predict(obs, deterministic=True)
        rgb = env.grid_world.get_rgb()
        states.append(rgb.copy())
        obs, reward, done, info = vec_env.step(action)
        if done:
            break

    images = [Image.fromarray(state) for state in states]
    images = iter(images)

    image = next(images)
    image.save(
        f"gridworld_{task_name}.gif",
        format="GIF",
        save_all=True,
        append_images=images,
        loop=0,
        fps=2,
    )


if __name__ == "__main__":
    # TEST TERMINATION
    test_task("tasks/lava.txt", algorithm=PPO)
    test_task("tasks/exit.txt", algorithm=PPO)
    test_task("tasks/bait.txt", algorithm=PPO)
    test_task("tasks/door.txt", algorithm=A2C)
    test_task("tasks/portal.txt", algorithm=PPO)
    test_task("tasks/maze.txt", algorithm=PPO)
    # TEST SPEED
    test_speed("tasks/lava.txt")
    test_speed("tasks/exit.txt")
    test_speed("tasks/bait.txt")
    test_speed("tasks/door.txt")
    test_speed("tasks/portal.txt")
    test_speed("tasks/maze.txt")
    # TEST CORRECTNESS
    test_correctness("tasks/lava.txt")
    test_correctness("tasks/exit.txt")
    test_correctness("tasks/bait.txt")
    test_correctness("tasks/door.txt")
    test_correctness("tasks/portal.txt")
    test_correctness("tasks/maze.txt")
    # Write one trajectory to gif
    # write_gif("tasks/lava.txt", algorithm=PPO)
    # write_gif("tasks/exit.txt", algorithm=PPO)
    # write_gif("tasks/bait.txt", algorithm=PPO)
    # write_gif("tasks/door.txt", algorithm=A2C)
    # write_gif("tasks/portal.txt", algorithm=PPO)
    # write_gif("tasks/maze.txt", algorithm=PPO)
