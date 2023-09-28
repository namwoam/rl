import random
import numpy as np
import json

from DP_solver import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
LEARNING_RATE   = 0.01
EPSILON         = 0.3
NUM_STEP        = 3

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", traj_file: str = "traj.json"):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        traj_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=INIT_POS,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld):
    print("Run MC prediction.")
    prediction = MonteCarloPrediction(
        grid_world, discount_factor=DISCOUNT_FACTOR
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()


def run_TD_prediction(grid_world: GridWorld):
    print("Run TD(0) prediction.")
    prediction = TDPrediction(
        grid_world, discount_factor=DISCOUNT_FACTOR, learning_rate=LEARNING_RATE
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()


def run_NstepTD_prediction(grid_world: GridWorld):
    print("Run N-step TD prediction.")
    prediction = NstepTDPrediction(
        grid_world, discount_factor=DISCOUNT_FACTOR, learning_rate=LEARNING_RATE, num_step=NUM_STEP
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    grid_world = init_grid_world("maze.txt", "traj.json")

    run_MC_prediction(grid_world)
    run_TD_prediction(grid_world)
    run_NstepTD_prediction(grid_world)
