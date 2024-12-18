import pygame
import argparse
from stable_baselines3 import PPO
from swerve_wrapper import SwerveGameEnv

# Edit these paths to point to your best models
OFFENSE_MODEL_PATH = 'models/blue_clones/v7'  # Path to best offensive (blue) model
DEFENSE_MODEL_PATH = 'models/red_clones/v7'   # Path to best defensive (red) model

class ModelTester:
    def __init__(self, blue_model_path=None, red_model_path=None):
        self.env = SwerveGameEnv(render=True)
        
        # Load models
        if blue_model_path:
            self.blue_agent = PPO.load(blue_model_path)
            print(f"Loaded blue model from {blue_model_path}")
        
        if red_model_path:
            self.red_agent = PPO.load(red_model_path)
            print(f"Loaded red model from {red_model_path}")

    def run_episode(self, render=True):
        state = self.env.reset()
        done = False
        steps = 0
        episode_rewards = []
        
        while not done:
            # Process pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            
            # Blue's turn
            blue_action, _ = self.blue_agent.predict(state, deterministic=True)
            next_state, blue_reward, done, _ = self.env.step(blue_action, role='blue')  # Add role='blue'
            episode_rewards.append(('blue', blue_reward))
            
            if not done:
                # Red's turn
                red_action, _ = self.red_agent.predict(next_state, deterministic=True)
                next_state, red_reward, done, _ = self.env.step(red_action, role='red')  # Add role='red'
                episode_rewards.append(('red', red_reward))
            
            state = next_state
            steps += 1
            
            if render:
                self.env.render()
                
            # Print detailed rewards
            print("\nDetailed Rewards:")
            for role, reward in episode_rewards:
                print(f"{role}: {reward:.2f}")
        
        return {
            'steps': steps,
            'blue_reward': sum(r for role, r in episode_rewards if role == 'blue'),
            'red_reward': sum(r for role, r in episode_rewards if role == 'red')
        }


    def run_multiple_episodes(self, num_episodes=10, render=True):
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            result = self.run_episode(render)
            if result is None:
                break
            print(f"Steps: {result['steps']}")
            print(f"Blue Reward: {result['blue_reward']:.2f}")
            print(f"Red Reward: {result['red_reward']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Test trained Swerve Drive models')
    parser.add_argument('--blue', type=str, default=OFFENSE_MODEL_PATH,
                      help='Path to blue model file')
    parser.add_argument('--red', type=str, default=DEFENSE_MODEL_PATH,
                      help='Path to red model file')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                      help='Disable rendering')
    args = parser.parse_args()
    
    tester = ModelTester(args.blue, args.red)
    tester.run_multiple_episodes(args.episodes, render=not args.no_render)

if __name__ == "__main__":
    main()
