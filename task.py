import numpy as np
from physics_sim import PhysicsSim

# def squaredScore(score):
#     return np.sign(score) * np.square(score)

# def axisDistanceBetween(axisIndex, pointA, pointB, axisWeights, absolute):
#     distance = (pointA[axisIndex] - pointB[axisIndex]) * axisWeights[axisIndex]
    
#     if (absolute == True):
#         return abs(distance)
    
#     return distance

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Init pose
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 0., 0., 0., 0.])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 20.])

    def get_reward(self):
        return np.tanh(self.sim.pose[2] / self.target_pos[2])

#     def get_reward_bad_attempt_2(self):
#         multiplier = 1

#         if (self.sim.pose[2] > self.target_pos[2]):
#             multiplier = self.sim.pose[2]
        
#         return self.sim.pose[2] * multiplier
        
#     def get_reward_bad_attempt_1(self):
#         if self.sim.done and self.sim.runtime > self.sim.time:
#             # Crash check: if done is true before the runtime finished, the quadcopter has crashed the ground
#             # Returns negative value to give a lot of penalty to crashes
#             return -10
#         else:
#             # Sets up each axis reward relevance
#             axisRewardWeights = [0, 0, 1]

#             # Calculates the (weighted) scores by distance
#             xScore = axisDistanceBetween(0, self.sim.pose, self.init_pose, axisRewardWeights, True)
#             yScore = axisDistanceBetween(1, self.sim.pose, self.init_pose, axisRewardWeights, True)
#             zScore = axisDistanceBetween(2, self.sim.pose, self.init_pose, axisRewardWeights, False)
            
#             # Sums the squared scores to boost the reward
#             reward = squaredScore(xScore) + squaredScore(yScore) + squaredScore(zScore)

#             # Calculates the maximum (weighted) scores by distance (having the target as the maximum score)
#             maxXScore = axisDistanceBetween(0, self.target_pos, self.init_pose, axisRewardWeights, True)
#             maxYScore = axisDistanceBetween(1, self.target_pos, self.init_pose, axisRewardWeights, True)
#             maxZScore = axisDistanceBetween(2, self.target_pos, self.init_pose, axisRewardWeights, False)
            
#             # Mimics the reward calc and simulates the maximum possible reward
#             max_reward = squaredScore(maxXScore) + squaredScore(maxYScore) + squaredScore(maxZScore)

#             # Bounds the reward to a range that goes from (-max_reward, +max_reward) to (-1, 1)
#             if (reward < -max_reward):
#                reward = -max_reward

#             if (reward > max_reward):
#                reward = max_reward

#             # Scales the reward from (-max_reward, +max_reward) to (-1, 1)
#             return reward / max_reward
#             # return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        # Finishes the episode when all upcoming steps achieves the target
        # if (reward > (1000 * self.action_repeat)):
        #    done = True

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
