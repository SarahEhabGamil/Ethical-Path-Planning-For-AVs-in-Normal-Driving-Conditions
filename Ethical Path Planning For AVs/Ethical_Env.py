import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


class EthicalEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.shape = (23, 4)
        self.setup_plot()
        self.counter12 = []
        self.counter13 = []
        self.counter14 = []

        self.counter3 = []
        self.counter13 = []
        self.counter21 = []
        self.sizeX = 4  # angle-action
        self.sizeY = 24  # angle-action
        self.sizeXY = self.sizeX * self.sizeY  # angle-action

        self.max_action = 2
        self.saveLastPos = np.array([100, 100])

        self.viewer = None

        self.min_pos_x_robot = 0
        self.min_pos_y_robot = 0
        self.min_pos_x_goal = 0
        self.min_pos_y_goal = 0
        self.min_pos_x_diff = -3
        self.min_pos_y_diff = -23

        self.max_pos_x_robot = 3
        self.max_pos_y_robot = 23
        self.max_pos_x_goal = 3  # +1 e7tyaty
        self.max_pos_y_goal = 23  # +1 e7tyaty
        self.max_pos_x_diff = 3  # 3-0 = 3 +1 e7tyaty
        self.max_pos_y_diff = 23  # 9-0 = 9 + 1 e7tyaty

        self.shape = (self.sizeY, self.sizeX)

        high = np.array([23.0, 3.0, 23.0, 3.0, 23.0, 3.0, 23.0, 23.0], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0, 0.0, -23.0, -3.0, -23.0, -23.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

        self.leftt = random.randint(6, 7)
        self.rightt = random.randint(3, 4)

        self.Obstacle_positionR0 = 3  # [3 0]
        self.Obstacle_positionC0 = 0
        self.Obstacle_positionR1 = 3  # [3 3]
        self.Obstacle_positionC1 = 3
        self.Obstacle_positionR2 = 4  # [5 1]# awel 3 shaghalen folla
        self.Obstacle_positionC2 = 1
        self.Obstacle_positionR3 = 5  # [5 2]# awel 3 shaghalen folla
        self.Obstacle_positionC3 = 2
        self.Obstacle_positionR4 = 5  # [7 0]
        self.Obstacle_positionC4 = 3
        self.Obstacle_positionR5 = 7  # [7 3]
        self.Obstacle_positionC5 = 3

        self.counter_steps = 0

        # Traffic lights
        self.traffic_light_state = 'red'  # Start with a single global state
        self.traffic_light_cycle = {'green': 20, 'yellow': 2, 'red': 10}
        self.traffic_light_timer = self.traffic_light_cycle['red']

        self.small_car_positions = [
            {'row': 6, 'col': 0},
            {'row': 10, 'col': 4},
            {'row': 13, 'col': 2}
        ]

        self.big_car_positions = [
            {'row': 6, 'col': 3},
            {'row': 11, 'col': 0}
        ]
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    def step(self, u):
        self.penalties = 0
        self.reward = 0

        self.update_traffic_lights()
        current_light = self.get_traffic_light_state()

        _cliff = np.zeros(self.shape, dtype=int)
        _cliff = np.zeros(self.shape, dtype=int)
        counter = 0
        for i in range(0, self.sizeY):
            for j in range(0, self.sizeX):
                _cliff[i, j] = self.ObservationS[counter]
                counter = counter + 1

        u = np.clip(u, -self.max_action, self.max_action)[0]
        # print(u,"u in step")

        agent_pos = (self.state[0], self.state[1])

        # Handling small and big cars
        for small_car_pos in self.small_car_positions:
            for big_car_pos in self.big_car_positions:
                if agent_pos[0] == small_car_pos['row'] == big_car_pos['row']:
                    distance_to_small_car = abs(agent_pos[1] - small_car_pos['col'])
                    distance_to_big_car = abs(agent_pos[1] - big_car_pos['col'])
                    if distance_to_small_car == 1:
                        u = -1
                    elif distance_to_big_car == 1 and distance_to_small_car > 1:
                        preferred_direction = "right" if agent_pos[1] < small_car_pos['col'] else "left"
                        u = -2 if preferred_direction == "right" else 0
                        self.penalties -= 20
                        # done = True
                    elif distance_to_small_car < distance_to_big_car:
                        preferred_direction = "right" if agent_pos[1] < small_car_pos['col'] else "left"
                        u = -2 if preferred_direction == "right" else 0
                        self.reward += 10

                    elif distance_to_big_car < distance_to_small_car:
                        preferred_direction = "right" if agent_pos[1] < big_car_pos['col'] else "left"
                        u = -2 if preferred_direction == "right" else 0
                        self.penalties -= 20

                        # done = True
                    else:
                        u = -1

        # Handling traffic lights
        if current_light == 'red' and u != 0:
            u = 2
            self.penalties -= 20
            # done = True
        else:
            self.reward += 5

        delta = [0, 0]
        # if action == 0 :#up
        #     delta = [0,0]
        if (u >= -2 and u < -1):
            # print("right")# right
            delta = [0, 1]

        elif (u >= -1 and u < 0):  # down
            # print("down")
            delta = [1, 0]

        elif (u >= 0 and u < 1):  # left
            # print("left")
            delta = [0, -1]
        elif (u >= 1 and u <= 2):  # stop
            # print("stop")
            delta = [0, 0]
        else:
            print("Error: Action out of range")

        # th, thdot = self.state  # th := theta

        Lrobot_positionR = self.state[0]
        Lrobot_positionC = self.state[1]

        Lgoal_positionR = self.state[2]
        Lgoal_positionC = self.state[3]
        self.counter_steps = self.counter_steps + 1


        self.last_u = u  # for rendering
        TempPosR = Lrobot_positionR
        TempPosC = Lrobot_positionC
        Lrobot_positionR = Lrobot_positionR + np.array(delta)[0]
        Lrobot_positionC = Lrobot_positionC + np.array(delta)[1]
        new_position = self._limit_coordinates(np.array([Lrobot_positionR, Lrobot_positionC])).astype(float)

        is_close, closest_obstacle = self.is_too_close(new_position)
        if is_close:
            self.penalties -= 5
        else :
            self.reward += 10
        Lrobot_positionR = new_position[0]
        Lrobot_positionC = new_position[1]


        Ddone = False
        DDD = False

        d = 1
        r = 0
        if _cliff[int(Lrobot_positionR), int(Lrobot_positionC)] == 4:

            Lrobot_positionR = TempPosR
            Lrobot_positionC = TempPosC


        Ldiff_y = (Lgoal_positionR) - (Lrobot_positionR)
        Ldiff_x = (Lgoal_positionC) - (Lrobot_positionC)
        if (Ldiff_y == 0 and Ldiff_x == 0):
            DDD = True
            self.reward+=500
            print("rewardd takenn")
            # r = -100
        costs = self.penalties +self.reward

        self.saveLastPos[0] = Ldiff_y
        self.saveLastPos[1] = Ldiff_x
        self.state = np.array([Lrobot_positionR, Lrobot_positionC, Lgoal_positionR, Lgoal_positionC, Ldiff_y, Ldiff_x,
                               self.rightt - Lrobot_positionR, self.leftt - Lrobot_positionR])
        return self.state, costs, DDD, {}

    def reset(self):
        print("Last Error: ", self.saveLastPos, "No of Steps: ", self.counter_steps)

        Lrobot_positionR = 0
        Lrobot_positionC = 1  # random.randint(0, 1)
        Lgoal_positionR = 23
        Lgoal_positionC = 1  # random.randint(0, 1)
        Ldiff_y = (Lgoal_positionR) - (Lrobot_positionR)
        Ldiff_x = (Lgoal_positionC) - (Lrobot_positionC)

        _cliff = np.zeros(self.shape, dtype=int)
        _cliff[Lgoal_positionR, Lgoal_positionC] = 5
        _cliff[Lrobot_positionR, Lrobot_positionC] = 1


        if self.rightt == 13:
            print("                                         Scenario 1 Activated")
            self.counter13.append(self.counter_steps)

        if self.rightt == 21:
            print("                                         Scenario 2 Activated")
            self.counter21.append(self.counter_steps)

        if self.leftt == 3:
            print("                                         Scenario 3 Activated")
            self.counter3.append(self.counter_steps)

        self.rightt = random.randint(13, 22)  # 17-21
        self.leftt = random.randint(3, 12)  # 7, 14
        self.counter_steps = 0

        # self.Obstacle_positionR0 = 3  # [3 0]
        # self.Obstacle_positionC0 = 0
        # self.Obstacle_positionR1 = 3#[3 3]
        # self.Obstacle_positionC1 = 3
        # self.Obstacle_positionR2 = self.leftt  # [5 1]# awel 3 shaghalen folla
        # self.Obstacle_positionC2 = 0
        # self.Obstacle_positionR3 = self.leftt  # [5 2]# awel 3 shaghalen folla
        # self.Obstacle_positionC3 = 1
        # self.Obstacle_positionR4 = self.rightt
        # self.Obstacle_positionC4 = 2
        # self.Obstacle_positionR5 = self.rightt
        # self.Obstacle_positionC5 = 3

        self.Obstacle_positionR2 = 3
        self.Obstacle_positionC2 = 0
        self.Obstacle_positionR3 = 21
        self.Obstacle_positionC3 = 2
        self.Obstacle_positionR4 = 6
        self.Obstacle_positionC4 = 2
        self.Obstacle_positionR5 = 21
        self.Obstacle_positionC5 = 3

        # _cliff[self.Obstacle_positionR0, self.Obstacle_positionC0] = 4
        # _cliff[self.Obstacle_positionR1, self.Obstacle_positionC1] = 4
        _cliff[self.Obstacle_positionR2, self.Obstacle_positionC2] = 4
        _cliff[self.Obstacle_positionR3, self.Obstacle_positionC3] = 4
        _cliff[self.Obstacle_positionR4, self.Obstacle_positionC4] = 4
        _cliff[self.Obstacle_positionR5, self.Obstacle_positionC5] = 4

        self.small_car_positions = [
            {'row': 13, 'col': 0},
            {'row': 12, 'col': 1},
            {'row': 14, 'col': 2}
        ]
        self.big_car_positions = [
            {'row': 13, 'col': 3},
            {'row': 16, 'col': 1},
            {'row': 17, 'col': 3}
        ]
        for car in self.small_car_positions:
            _cliff[car['row'], car['col']] = 4
        for car in self.big_car_positions:
            _cliff[car['row'], car['col']] = 4

        # self.ObservationS = np.array([_cliff[0,0], _cliff[0,1], _cliff[0,2], _cliff[0,3], _cliff[1,0], _cliff[1,1], _cliff[1,2], _cliff[1,3], _cliff[2,0], _cliff[2,1], _cliff[2,2], _cliff[2,3],        _cliff[3,0] ,_cliff[3,1]        ,_cliff[3,2]   ,_cliff[3,3]        ,_cliff[4,0]         ,_cliff[4,1]         ,_cliff[4,2]      ,_cliff[4,3]         ,_cliff[5,0]         ,_cliff[5,1]         ,_cliff[5,2]   ,_cliff[5,3]         ,_cliff[6,0]         ,_cliff[6,1]         ,_cliff[6,2]  ,_cliff[6,3]         ,_cliff[7,0]         ,_cliff[7,1]         ,_cliff[7,2]  ,_cliff[7,3]         ,_cliff[8,0]         ,_cliff[8,1]         ,_cliff[8,2],_cliff[8,3] ,_cliff[9,0],_cliff[9,1],_cliff[9,2],_cliff[9,3]])
        self.ObservationS = np.array([], dtype=int)
        for i in range(0, self.sizeY):
            for j in range(0, self.sizeX):
                self.ObservationS = np.append(self.ObservationS, _cliff[i, j])

        OoO = 0

        self.state = np.array([Lrobot_positionR, Lrobot_positionC, Lgoal_positionR, Lgoal_positionC, Ldiff_y, Ldiff_x,
                               self.rightt - Lrobot_positionR, self.leftt - Lrobot_positionR])
        self.last_u = None
        return self.state

    def update_traffic_lights(self):
        self.traffic_light_timer -= 1  # Decrement the timer
        if self.traffic_light_timer <= 0:
            if self.traffic_light_state == 'green':
                next_state = 'yellow'
            elif self.traffic_light_state == 'yellow':
                next_state = 'red'
            elif self.traffic_light_state == 'red':
                next_state = 'green'

            # Update to the next state and reset the timer
            self.traffic_light_state = next_state
            self.traffic_light_timer = self.traffic_light_cycle[next_state]
    def get_traffic_light_state(self):
        return self.traffic_light_state
    def is_too_close(self, position):
        proximity_threshold = 0.9
        obstacles = [
            (self.Obstacle_positionR0, self.Obstacle_positionC0),
            (self.Obstacle_positionR1, self.Obstacle_positionC1),
            (self.Obstacle_positionR2, self.Obstacle_positionC2),
            (self.Obstacle_positionR3, self.Obstacle_positionC3),
            (self.Obstacle_positionR4, self.Obstacle_positionC4),
            (self.Obstacle_positionR5, self.Obstacle_positionC5)
        ]

        for obs_r, obs_c in obstacles:
            if abs(position[0] - obs_r) <= proximity_threshold and abs(position[1] - obs_c) <= proximity_threshold:
                # print("got too close")
                return True, (obs_r, obs_c)
        return False, None
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 9))  # Increase figure height
        self.grid = np.zeros(self.shape)
        self.img = self.ax.imshow(self.grid, cmap=ListedColormap(['white', 'black', '#CD402D', '#6A9CD1', '#0E5199', '#379F32']),
                                  vmin=0, vmax=5)

        # Set up the traffic light above the grid
        self.traffic_light_circle = patches.Circle((self.shape[1] // 2, -1), radius=1, color='gray',
                                                   transform=self.ax.transData)  # Adjust radius and position
        self.ax.add_patch(self.traffic_light_circle)

        # Adjust grid aesthetics
        self.ax.set_xticks(np.arange(-.5, self.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.shape[0], 1), minor=True)
        self.ax.grid(which='minor', linestyle='-', color='w', linewidth=2)
        self.ax.tick_params(which="minor", size=0)

    def update_plot(self, state):
        self.state = state
        self.grid = np.zeros(self.shape)

        # Update agent's position
        agent_row, agent_col = int(self.state[0]), int(self.state[1])
        self.grid[agent_row, agent_col] = 1  # Agent's position marked blue

        # Update traffic light display based on its state
        traffic_light_colors = {'green': 'green', 'yellow': 'yellow', 'red': 'red'}
        self.traffic_light_circle.set_color(traffic_light_colors[self.get_traffic_light_state()])

        # Update obstacles
        obstacles_positions = [
            (self.Obstacle_positionR0, self.Obstacle_positionC0),
            (self.Obstacle_positionR1, self.Obstacle_positionC1),
            (self.Obstacle_positionR2, self.Obstacle_positionC2),
            (self.Obstacle_positionR3, self.Obstacle_positionC3),
            (self.Obstacle_positionR4, self.Obstacle_positionC4),
            (self.Obstacle_positionR5, self.Obstacle_positionC5),
        ]
        for pos in obstacles_positions:
            self.grid[pos[0], pos[1]] = 2  # Obstacles marked red

        # Update small and big cars
        for small_car_pos in self.small_car_positions:
            self.grid[small_car_pos['row'], small_car_pos['col']] = 3  # Small car marked yellow
        for big_car_pos in self.big_car_positions:
            self.grid[big_car_pos['row'], big_car_pos['col']] = 4  # Big car marked purple

        # Goal position (if applicable)
        goal_row, goal_col = int(self.state[2]), int(self.state[3])
        self.grid[goal_row, goal_col] = 5  # Goal's position marked green

        self.img.set_data(self.grid)
        return (self.img, self.traffic_light_circle)

    def close(self):
        if self.viewer:
            plt.close(self.fig)
            self.viewer = None

