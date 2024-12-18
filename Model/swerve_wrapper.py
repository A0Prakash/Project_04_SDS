import gym
import numpy as np
from gym import spaces
from constants import *
from robot import Robot
from game_functions import *

class SwerveGameEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.current_role = 'blue'  # Default role
        self.should_render = render
        
        # Initialize pygame and Box2D world
        pygame.init()
        if self.should_render:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))  # Off-screen surface when not rendering
        self.clock = pygame.time.Clock()
        
        # Define action spaces for swerve drive
        # Actions: [vx, vy, omega] each normalized between -1 and 1
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        # Define observation space
        # [robot_x, robot_y, robot_angle, robot_vx, robot_vy, robot_omega,
        #  opponent_x, opponent_y, opponent_angle, opponent_vx, opponent_vy, opponent_omega]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, -20, -20, -10, 
                         0, 0, -np.pi, -20, -20, -10]),
            high=np.array([WINDOW_WIDTH, WINDOW_HEIGHT, np.pi, 20, 20, 10,
                         WINDOW_WIDTH, WINDOW_HEIGHT, np.pi, 20, 20, 10]),
            dtype=np.float32
        )
        
        self.reset()

    def normalize_state(self, state):
        """Normalize state to be between -1 and 1"""
        return 2 * (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low) - 1

    def set_role(self, role):
        """Set the current role for the environment"""
        if role not in ['blue', 'red']:
            raise ValueError("Role must be 'blue' or 'red'")
        self.current_role = role

    def get_state(self, robot, opponent):
        """Convert game state to observation vector"""
        if self.current_role == 'red':
            robot, opponent = opponent, robot  # Swap robots based on role
            
        raw_state = np.array([
            robot.body.position.x * PPM,
            robot.body.position.y * PPM,
            robot.body.angle,
            robot.body.linearVelocity.x,
            robot.body.linearVelocity.y,
            robot.body.angularVelocity,
            opponent.body.position.x * PPM,
            opponent.body.position.y * PPM,
            opponent.body.angle,
            opponent.body.linearVelocity.x,
            opponent.body.linearVelocity.y,
            opponent.body.angularVelocity
        ], dtype=np.float32)
        
        # Clip values to be within observation space bounds
        raw_state = np.clip(raw_state, self.observation_space.low, self.observation_space.high)
        return self.normalize_state(raw_state)

    def reset(self):
        """Reset the environment to initial state"""
        # Create/reset Box2D world
        self.world = b2World(gravity=(0, 0), doSleep=True)
        
        # Create walls with proper physics properties
        create_walls(self.world)
        
        # Reset robots to starting positions with correct colors
        self.robot_red = Robot(WINDOW_WIDTH//2, RED_START_Y, (200, 30, 30), self.world)
        self.robot_blue = Robot(WINDOW_WIDTH//2, BLUE_START_Y, (30, 30, 200), self.world)
        
        # Reset velocities and orientations
        self.robot_red.body.linearVelocity = b2Vec2(0, 0)
        self.robot_red.body.angularVelocity = 0
        self.robot_red.body.angle = 0
        
        self.robot_blue.body.linearVelocity = b2Vec2(0, 0)
        self.robot_blue.body.angularVelocity = 0
        self.robot_blue.body.angle = 0
        
        # Reset swerve modules
        for module in self.robot_red.modules:
            module.wheel_angle = 0
            module.wheel_speed = 0
            module.last_wheel_angle = 0
        
        for module in self.robot_blue.modules:
            module.wheel_angle = 0
            module.wheel_speed = 0
            module.last_wheel_angle = 0
        
        self.game_start_time = pygame.time.get_ticks()
        self.steps = 0
        self.progress = 0
        
        if self.current_role == 'blue':
            return self.get_state(self.robot_blue, self.robot_red)
        else:
            return self.get_state(self.robot_red, self.robot_blue)

    def step(self, action, role=None):
        """Take a step in the environment"""
        if role is None:
            role = self.current_role
            
        self.steps += 1
        
        # Ensure action is a numpy array and has the right shape
        action = np.array(action).flatten()
        
        # Scale actions to actual robot velocities
        vx, vy, omega = action * [self.robot_blue.max_speed, 
                                self.robot_blue.max_speed,
                                self.robot_blue.max_omega]
        
        # Apply action to appropriate robot
        if role == 'blue':
            self.robot_blue.apply_movement(vx, vy, omega, field_oriented=True, dt=1/60.0)
        else:
            self.robot_red.apply_movement(vx, vy, omega, field_oriented=True, dt=1/60.0)
        
        # Step physics
        self.world.Step(1/60.0, 8, 3)
        
        # Get new state based on current role
        if role == 'blue':
            new_state = self.get_state(self.robot_blue, self.robot_red)
        else:
            new_state = self.get_state(self.robot_red, self.robot_blue)
        
        # Calculate reward
        reward = self._calculate_reward(role)
        
        # Check if done
        done = self._is_done()
        
        if self.should_render:
            self.render()
        
        return new_state, reward, done, {}

    def _calculate_reward(self, role):
        """Calculate reward based on role"""
        blue_top = self.robot_blue.body.position.y * PPM - (self.robot_blue.size * PPM / 2)
        blue_start_y = BLUE_START_Y
        timeout = self.steps >= GAME_DURATION * 60
        blue_scored = blue_top < FIELD_MARGIN
        
        if role == 'blue':
            # Time penalty (-1 per second at 60 FPS)
            time_penalty = -1/60
            
            # Calculate forward progress in inches
            progress_inches = (blue_start_y - blue_top) / PPM * 39.3701
            progress_reward = max(0, progress_inches * 0.1)
            
            if progress_reward > self.progress:
                self.progress = progress_reward
            else:
                progress_reward = 0
            
            if blue_scored:
                reward = 50 + progress_reward + (time_penalty * self.steps)
                print(f"Blue scored! Reward: {reward}")
                return reward
            elif timeout:
                reward = -50 + progress_reward + (time_penalty * self.steps)
                print(f"Blue lost by timeout! Reward: {reward}")
                return reward
            else:
                return progress_reward + time_penalty
        else:  # red
            # Distance-based defense reward
            distance_to_blue = abs(self.robot_red.body.position.y - self.robot_blue.body.position.y) * PPM
            optimal_distance = ROBOT_SIZE_PIXELS * 1.5  # Slightly larger than robot size
            distance_reward = max(0, 1 - abs(distance_to_blue - optimal_distance) / (ROBOT_SIZE_PIXELS * 2))
            
            # Position-based reward (encourage staying between blue and goal)
            defensive_position = (self.robot_red.body.position.y > self.robot_blue.body.position.y)
            position_reward = 0.5 if defensive_position else -0.5
            
            # Combine continuous rewards
            continuous_reward = (distance_reward + position_reward) * 0.1
            
            if blue_scored:
                print("Red lost - Blue scored")
                return -50 + continuous_reward
            elif timeout:
                print("Red won by timeout!")
                return 50 + continuous_reward
            else:
                return continuous_reward

    def _is_done(self):
        """Check if episode is done"""
        blue_top = self.robot_blue.body.position.y * PPM - (self.robot_blue.size * PPM / 2)
        return blue_top < FIELD_MARGIN or self.steps >= GAME_DURATION * 60

    def render(self, mode='human'):
        """Render the environment"""
        if not self.should_render:
            return
        if mode != 'human':
            return
            
        # Fill screen with white
        self.screen.fill((255, 255, 255))
        
        # Draw field
        pygame.draw.rect(self.screen, FIELD_COLOR, 
                        (FIELD_MARGIN, FIELD_MARGIN, 
                        FIELD_WIDTH, FIELD_HEIGHT))
        
        # Draw goal area with stripes
        draw_goal_area(self.screen)
        
        # Draw walls
        pygame.draw.line(self.screen, BOUNDARY_COLOR,
                        (FIELD_MARGIN, FIELD_MARGIN + FIELD_HEIGHT),
                        (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN + FIELD_HEIGHT),
                        BOUNDARY_THICKNESS)
        pygame.draw.line(self.screen, BOUNDARY_COLOR,
                        (FIELD_MARGIN, FIELD_MARGIN),
                        (FIELD_MARGIN, FIELD_MARGIN + FIELD_HEIGHT),
                        BOUNDARY_THICKNESS)
        pygame.draw.line(self.screen, BOUNDARY_COLOR,
                        (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN),
                        (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN + FIELD_HEIGHT),
                        BOUNDARY_THICKNESS)
        
        # Draw robots
        self.robot_red.draw(self.screen)
        self.robot_blue.draw(self.screen)
        
        # Draw timer
        draw_timer(self.screen, self.game_start_time)
        
        # Draw game over message if game is over
        if self._is_done():
            blue_top = self.robot_blue.body.position.y * PPM - (self.robot_blue.size * PPM / 2)
            if blue_top < FIELD_MARGIN:
                winner_message = "BLUE WINS!"
            elif self.steps >= GAME_DURATION * 60:
                winner_message = "RED WINS!"
            draw_game_over(self.screen, winner_message)
        
        pygame.display.flip()
        self.clock.tick(60)
