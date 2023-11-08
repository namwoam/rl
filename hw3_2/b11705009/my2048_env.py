from __future__ import print_function

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO
import sys


def count_same_value_neighbors(matrix, i, j):
    element_value = matrix[i, j]
    count = 0

    # Define the indices of the neighboring elements
    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    for neighbor_i, neighbor_j in neighbors:
        # Check if the neighboring indices are within the valid range
        if 0 <= neighbor_i < matrix.shape[0] and 0 <= neighbor_j < matrix.shape[1]:
            if matrix[neighbor_i, neighbor_j] == element_value:
                count += 1

    return count


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class IllegalMove(Exception):
    pass


def stack(flat, layers=16):
    """Convert an [4, 4] representation into [layers, 4, 4] with one layers for each value."""
    # representation is what each layer represents
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:, :, np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)
    layered = np.transpose(layered, (2, 0, 1))
    return layered


class My2048Env(gym.Env):
    metadata = {
        "render_modes": ['ansi', 'human', 'rgb_array'],
        "render_fps": 2,
    }

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size
        self.illegal_count = 10

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Foul counts for illegal moves
        self.foul_count = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(
            0, 1, (layers, self.w, self.h), dtype=int)

        # TODO: Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-100)

        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
            'highest': 0,
            'score': 0,
        }
        try:
            pre_state = self.Matrix.copy()
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)

            # TODO: Add reward according to weighted states (optional)
            weight = np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0.1, 0.1],
                [0, 0, 0.1, 0.3]])
            reward += np.sum(np.dot(weight, self.Matrix))

            neighbor_count_matrix = np.zeros_like(self.Matrix, dtype=int)

            for i in range(self.Matrix.shape[0]):
                for j in range(self.Matrix.shape[1]):
                    neighbor_count_matrix[i, j] = count_same_value_neighbors(
                        self.Matrix, i, j)

            reward += np.sum(neighbor_count_matrix)*0.4

        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            reward = self.illegal_move_reward

            # TODO: Modify this part for the agent to have a chance to explore other actions (optional)
            if self.illegal_count > 0:
                self.illegal_count -= 1
            else:
                done = True

        truncate = False
        info['highest'] = self.highest()
        info['score'] = self.score

        # Return observation (board state), reward, done, truncate and info dict
        return stack(self.Matrix), reward, done, truncate, info

    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0
        self.foul_count = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix), {}

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        # 0 for towards up left, 1 for towards bottom right
        shift_direction = dir_mod_two ^ dir_div_two

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board
