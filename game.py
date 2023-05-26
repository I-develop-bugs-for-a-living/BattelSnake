import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 300

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class SnakeGameAI:
    def _is_moving_towards_tail(self, action):
        # Calculate the next head position based on the action
        if len(self.snake) < 7:
            return False
        if np.array_equal(action, [1, 0, 0]):
            new_head = Point(self.head.x + BLOCK_SIZE, self.head.y)
        elif np.array_equal(action, [0, 1, 0]):
            new_head = Point(self.head.x, self.head.y + BLOCK_SIZE)
        else:
            new_head = Point(self.head.x - BLOCK_SIZE, self.head.y)  # Assuming LEFT action

        # Check if the new head position is equal to the position of the tail segment
        if new_head == self.snake[-1]:
            return True

        # Check if the distance between the head and the tail is at most 4 blocks
        if len(self.snake) > 1:
            tail = self.snake[-1]
            distance = abs(new_head.x - tail.x) + abs(new_head.y - tail.y)
            if distance <= 4:
                return True

        return False
    def is_head_adjacent_to_body(self):
        # Get the head's adjacent points
        adjacents = [
            Point(self.head.x - BLOCK_SIZE, self.head.y), # left
            Point(self.head.x + BLOCK_SIZE, self.head.y), # right
            Point(self.head.x, self.head.y - BLOCK_SIZE), # up
            Point(self.head.x, self.head.y + BLOCK_SIZE)  # down
        ]

        # Check if any adjacent points are in the snake's body
        for pt in adjacents:
            if pt in self.snake[1:]: # Exclude the head itself
                return True

        return False


    def __init__(self, w=440, h=440):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.BLOCK_SIZE = BLOCK_SIZE


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -6 if self.isCollidingWithSelf() else -4 if self.isCollidingWithWall else -4
            return reward, game_over, self.score
        
        if self._is_moving_towards_tail(action):
            reward = -2
            print("Moving towards tail")
        elif self.is_head_adjacent_to_body():
            reward = 0

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward += 7
            self._place_food()
        else:
            self.snake.pop()
        reward = clamp(reward, -200, 80)
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def isCollidingWithSelf(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        return False
    def isCollidingWithWall(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        return False

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)