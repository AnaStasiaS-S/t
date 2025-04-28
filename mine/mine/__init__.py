#!/usr/bin/env python3
import rclpy
from rclpy.publisher import Publisher
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import LaserScan
from flatland_msgs.srv import MoveModel
from flatland_msgs.msg import Collisions
from typing import Tuple, Callable

from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

import numpy as np
from torch import nn
import torch as th
import time
import threading

class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
    
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        f_layer_dim_pi: int = 64,
        f_layer_dim_vf: int = 64,
        m_layer_dim_pi: int = 64,
        m_layer_dim_vf: int = 64,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, f_layer_dim_pi), nn.LeakyReLU(),
            ResNet(nn.Sequential(                
                nn.Linear(f_layer_dim_pi, m_layer_dim_pi), nn.ReLU()  ,
                # nn.Dropout(0.2),
                nn.BatchNorm1d(m_layer_dim_pi),
                nn.Linear(m_layer_dim_pi, m_layer_dim_pi), nn.ReLU()
            )),
                nn.Linear(m_layer_dim_pi, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, f_layer_dim_vf), nn.LeakyReLU(),
            ResNet(nn.Sequential(                
                nn.Linear(f_layer_dim_vf, m_layer_dim_vf), nn.ReLU()  ,
                nn.Dropout(0.2),
                nn.BatchNorm1d(m_layer_dim_vf),
                nn.Linear(m_layer_dim_vf, m_layer_dim_vf), nn.ReLU()
            )),
                nn.Linear(m_layer_dim_vf, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

class SerpControllerEnv(Node, Env):
    def __init__(self) -> None:
        super().__init__("SerpControllerEnv")

        # Predefined speed for the robot
        linear_speed = 0.5
        angular_speed = 1.57079632679

        # Set of actions. Defined by their linear and angular speed
        self.actions = [(linear_speed, 0.0), # move forward
                        (0.0, angular_speed), # rotate left
                        (0.0, -angular_speed)] # rotate right

        # How close the robot needs to be to the target to finish the task
        self.end_range = 0.2

        # Number of divisions of the LiDAR
        self.n_lidar_sections = 9
        self.lidar_sample = []

        # Variables that track a possible end state
        # current distance to target
        self.distance_to_end = 10.0
        # true if a collision happens
        self.collision = False

        # Possible starting positions
        self.start_positions = [(0.0, 0.0, 1.57079632679), (1.6, 1.6, 3.14159265359)]
        # self.start_positions = [(0.0, 0.0, 1.57079632679)]
        # Current position
        self.position = 0

        self.step_number = 0

        # Maximum number of steps before it times out
        self.max_steps = 200

        # Records previous action taken. At the start of an episode, there is no prior action so -1 is assigned
        self.previous_action = -1

        # Used for data collection during training
        self.total_step_cnt = 0
        self.total_episode_cnt = 0
        self.training = False
        self.rew = 0
                                    
        # **** Create publishers ****
        self.pub:Publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        # ***************************

        # **** Create subscriptions ****
        self.create_subscription(LaserScan, "/static_laser", self.processLiDAR, 1)

        self.create_subscription(LaserScan, "/end_beacon_laser", self.processEndLiDAR, 1)

        self.create_subscription(Collisions, "/collisions", self.processCollisions, 1)
        # ******************************

        # **** Define action and state spaces ****

        # action is an integer between 0 and 2 (total of 3 actions)
        self.action_space = Discrete(len(self.actions))
        # state is represented by a numpy.Array with size 9 and values between 0 and 2
        self.observation_space = Box(0, 2, shape=(self.n_lidar_sections,), dtype=np.float64)

        # ****************************************

        # Initial state
        self.state = np.array(self.lidar_sample)

    # Resets the environment to an initial state
    def reset(self):
        # Make sure the robot is stopped
        self.change_robot_speeds(self.pub, 0.0, 0.0)

        if self.total_step_cnt != 0: 
            self.total_episode_cnt += 1

        
        # **** Move robot and end beacon to new positions ****
        start_pos = self.start_positions[self.position]
        # self.position = 1 - self.position       # changes index visa versa to change beacon and cart
        end_pos = self.start_positions[1 - self.position]
        
        self.move_model('mine', start_pos[0], start_pos[1], start_pos[2])
        self.move_model('end_beacon', end_pos[0], end_pos[1], 0.0)
        # ****************************************************

        # **** Reset necessary values ****
        if start_pos == self.start_positions[self.position]: # zero total reward on episode start
            self.rew = 0
        self.lidar_sample = []
        self.wait_lidar_reading()
        self.state = np.array(self.lidar_sample)

        # Flatland can sometimes send several collision messages. 
        # This makes sure that no collisions are wrongfully detected at the start of an episode 
        time.sleep(0.1)

        self.distance_to_end = 10.0
        self.collision = False
        self.step_number = 0
        self.previous_action = -1
        # ********************************

        return self.state

    # Performs a step for the RL agent
    def step(self, action): 

        # **** Performs the action and waits for it to be completed ****
        self.change_robot_speeds(self.pub, self.actions[action][0], self.actions[action][1])

        self.lidar_sample = []
        self.wait_lidar_reading()
        self.change_robot_speeds(self.pub, 0.0, 0.0)
        # **************************************************************

        # Register current state
        self.state = np.array(self.lidar_sample)

        self.step_number += 1
        self.total_step_cnt += 1

        # **** Calculates the reward and determines if an end state was reached ****
        done = False

        end_state = ''

        if self.collision:
            end_state = "colision"
            done = True
            if self.distance_to_end >= 1.2:
                reward = -200
            else:
                reward = -200 + int(round(self.distance_to_end, 1))*10
            
        elif self.distance_to_end < self.end_range:
            end_state = "finished"
            reward = 500 + (200 - self.step_number)
            done = True
            # print("Disreward)
        elif self.step_number >= self.max_steps:
            end_state = "timeout"            
            done = True
            if self.distance_to_end >= 1.2:
                reward = -200
            else:
                reward = -200 + int(round(self.distance_to_end, 1))*10
        # elif action == 0:
            # reward = 2
        else:
            if self.distance_to_end >= 1.2:
                reward = 0
            else:
                reward = int(round(self.distance_to_end, 1))*10
        # **************************************************************************

        info = {'end_state': end_state}

        self.rew += reward

        if done and self.training:
            file = open("mine/logs/train.csv", "a")
            self.get_logger().info('Training - Episode ' + str(self.total_episode_cnt) + ' end state: ' + end_state + ' reward:' + str(self.rew))
            self.get_logger().info('Total steps: ' + str(self.total_step_cnt))
            self.get_logger().info('Steps in this episode: ' + str(self.step_number))
            file.write(f"{self.total_episode_cnt}, {self.rew}, {self.step_number}\n")

        return self.state, reward, done, info

    def render(self): pass

    def close(self): pass

    def reset_counters(self):
        self.total_step_cnt = 0
        self.total_episode_cnt = 0

    # Change the speed of the robot
    def change_robot_speeds(self, publisher, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        publisher.publish(twist_msg)

    # Waits for a new LiDAR reading.
    # A new LiDAR reading is also used to signal when an action should finish being performed.
    def wait_lidar_reading(self):
        while len(self.lidar_sample) != self.n_lidar_sections: pass

    # Send a request to move a model
    def move_model(self, model_name, x, y, theta):
        client = self.create_client(MoveModel, "/move_model")
        client.wait_for_service()
        request = MoveModel.Request()
        request.name = model_name
        request.pose = Pose2D()
        request.pose.x = x
        request.pose.y = y
        request.pose.theta = theta
        client.call_async(request)
    
    # Sample LiDAR data
    # Divite into sections and sample the lowest value from each
    def processLiDAR(self, data):
        self.lidar_sample = []

        rays = data.ranges
        rays_per_section = len(rays) // self.n_lidar_sections

        for i in range(self.n_lidar_sections - 1):
            self.lidar_sample.append(min(rays[rays_per_section * i:rays_per_section * (i + 1)]))
        self.lidar_sample.append(min(rays[(self.n_lidar_sections - 1) * rays_per_section:]))

    
    # Handle end beacon LiDAR data
    # Lowest value is the distance from robot to target
    def processEndLiDAR(self, data):
        clean_data = [x for x in data.ranges if str(x) != 'nan']
        # print("clean_data", clean_data)
        if not clean_data: return
        self.distance_to_end = min(clean_data)
        # print("distance_to_end", self.distance_to_end)
    
    # Process collisions
    def processCollisions(self, data):
        if len(data.collisions) > 0:
            self.collision = True

    # Run an entire episode manually for testing purposes
    # return true if succesful
    def run_episode(self, agent):
        
        com_reward = 0

        obs = self.reset()
        done = False
        while not done:
            action, states = agent.predict(obs)
            obs, rewards, done, info = self.step(action)
            com_reward += rewards

        #file_test = open("mine/logs/test.csv", "a")
        
        self.get_logger().info('Episode concluded. End state: ' + info['end_state'] + '  Commulative reward: ' + str(com_reward))
        #file_test.write(f"{com_reward}\n")

        return info['end_state'] == 'finished'

    def run_rl_alg(self):   
        file_test = open("mine/logs/test.csv", "a")

        check_env(self)
        
       
        self.wait_lidar_reading()   

        # Create the agent
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        agent = PPO("MlpPolicy", self, verbose=1, policy_kwargs=policy_kwargs)
        # agent = PPO(CustomActorCriticPolicy, self, verbose=1)

        # Target accuracy
        min_accuracy = 0.8
        # Current accuracy
        accuracy = 0
        # Number of tested episodes in each iteration
        n_test_episodes = 10

        training_iterations = 0
        
        
        while accuracy < min_accuracy:
            training_steps= 3000
            self.get_logger().info('Starting training for ' + str(training_steps) + ' steps')
            # file.write('Starting training for ' + str(training_steps) + ' steps')

            self.training = True
            self.reset_counters()

            # Train the agent            
            agent.learn(total_timesteps=training_steps)
            

            self.training = False

            successful_episodes = 0

            # Test the agent
            for i in range(n_test_episodes):
                self.get_logger().info('Testing episode number ' + str(i + 1) + '.')
                #file_test.write(f"Test episode {i+1}. Reward: ")
                if self.run_episode(agent): successful_episodes += 1
            
            # Calculate the accuracy
            accuracy = successful_episodes/n_test_episodes

            self.get_logger().info('Testing finished. Accuracy: ' + str(accuracy))
            file_test.write(f'Testing accuracy: {accuracy}\n')

            training_iterations += 1

        self.get_logger().info('Training Finished. Training iterations: ' + str(training_iterations) + '  Accuracy: ' + str(accuracy))
        file_test.write(f"Training Finished. Training iterations: {training_iterations}. Accuracy: {accuracy}\n")
        file_test.close()

        agent.save("mine/ppo")

def main(args = None):
    rclpy.init()
    
    mine = SerpControllerEnv()

    thread = threading.Thread(target=mine.run_rl_alg)
    thread.start()

    rclpy.spin(mine)



if __name__ == "__main__":
    main()
