import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from swerve_wrapper import SwerveGameEnv
import random
from constants import *

class MetricCallback(BaseCallback):
    """Custom callback for collecting episode metrics"""
    def __init__(self, role='blue', verbose=0):
        super().__init__(verbose)
        self.episode_info = []
        self.episode_length = 0
        self.role = role
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_length += 1
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        
        if self.locals['dones'][0]:
            if self.role == 'red':
                won = (reward >= 50)  # Red wins on timeout
                result = "timeout win" if won else "blue scored"
            else:  # blue
                won = (reward > 0)  # Blue wins on scoring
                result = "scored" if won else "timeout loss"
            
            self.episode_info.append({
                'reward': self.current_episode_reward,
                'length': self.episode_length,
                'won': won,
                'result': result
            })
            
            # Reset episode tracking
            self.episode_length = 0
            self.current_episode_reward = 0
        return True

    def get_episode_statistics(self):
        if not self.episode_info:
            return {
                'win_rate': 0.0,
                'avg_reward': 0.0,
                'avg_length': 0.0,
                'n_episodes': 0,
                'wins': 0,
                'total': 0
            }
            
        n_episodes = len(self.episode_info)
        wins = sum(1 for ep in self.episode_info if ep['won'])
        total_reward = sum(ep['reward'] for ep in self.episode_info)
        
        return {
            'win_rate': (wins / n_episodes),
            'avg_reward': total_reward / n_episodes,
            'avg_length': sum(ep['length'] for ep in self.episode_info) / n_episodes,
            'n_episodes': n_episodes,
            'wins': wins,
            'total': n_episodes
        }

class SelfPlayPPOTrainer:
    def __init__(self, save_dir="models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create clone directories
        os.makedirs(f"{save_dir}/blue_clones", exist_ok=True)
        os.makedirs(f"{save_dir}/red_clones", exist_ok=True)
        
        # Initialize clone pools
        self.blue_clones = []  # List of (version, model_path)
        self.red_clones = []
        self.window_size = 5  # Keep last 5 versions
        
        # Create environments
        self.env = DummyVecEnv([lambda: SwerveGameEnv(render=False)])
        self.eval_env = DummyVecEnv([lambda: SwerveGameEnv(render=False)])
        
        # Initialize agents with role-specific hyperparameters
        self.blue_agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # Moderate exploration for offensive behavior
            verbose=1
        )
        
        self.red_agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # Higher exploration for defensive behavior
            verbose=1
        )

    def _evaluate_model(self, model, role, num_episodes=5):
        """Evaluate a model's performance"""
        env = DummyVecEnv([lambda: SwerveGameEnv(render=False)])
        wins = 0
        
        for _ in range(num_episodes):
            env.envs[0].set_role(role)
            state = env.reset()
            done = [False]
            
            while not done[0]:
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, _ = env.step([action])
                if done[0] and ((role == 'blue' and reward > 0) or (role == 'red' and reward >= 50)):
                    wins += 1
                    
        return wins / num_episodes

    def add_clone(self, agent, clones_list, version, role):
        """Add new clone to window, removing oldest if necessary"""
        model_path = f"{self.save_dir}/{role}_clones/v{version}"
        agent.save(model_path)
        
        # Only add if performance improved or no existing clones
        if len(clones_list) == 0 or self._evaluate_model(agent, role) > self._evaluate_model(PPO.load(clones_list[-1][1]), role):
            clones_list.append((version, model_path))
            if len(clones_list) > self.window_size:
                clones_list.pop(0)
            print(f"Added new {role} clone version {version}")
        else:
            print(f"Skipped adding {role} clone version {version} due to poor performance")

    def load_opponent(self, clones_list, role):
        """Load random clone as opponent"""
        if not clones_list:
            # If no clones yet, create a new PPO agent
            return PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                verbose=0
            )
        # Randomly select a clone
        version, model_path = random.choice(clones_list)
        print(f"Loading {role} opponent version {version}")
        return PPO.load(model_path)

    def _adjust_opponent_difficulty(self, opponent, difficulty):
        """Adjust opponent's behavior based on difficulty level"""
        if difficulty == 'easy':
            # Reduce action magnitude for easier opponents
            def modified_predict(state, deterministic=False):
                action, states = opponent.predict(state, deterministic)
                return action * 0.5, states  # Reduce action magnitude
            opponent.predict = modified_predict
        elif difficulty == 'medium':
            # Use normal actions but with randomization
            def modified_predict(state, deterministic=False):
                action, states = opponent.predict(state, deterministic)
                noise = np.random.normal(0, 0.2, size=action.shape)
                return np.clip(action + noise, -1, 1), states
            opponent.predict = modified_predict
        else:  # 'hard'
            # Use original behavior
            pass

    def train_phase(self, training_role='blue', timesteps=100000):
        """Train one agent while using clones for opponent"""
        print(f"\nTraining {training_role.upper()} agent...")
        
        training_agent = self.blue_agent if training_role == 'blue' else self.red_agent
        opponent_clones = self.red_clones if training_role == 'blue' else self.blue_clones
        opponent_role = 'red' if training_role == 'blue' else 'blue'
        
        # Set the role in both environments
        self.env.envs[0].set_role(training_role)
        self.eval_env.envs[0].set_role(training_role)
        
        steps_per_opponent = timesteps // 5
        metrics = None
        
        for i in range(5):
            opponent = self.load_opponent(opponent_clones, opponent_role)
            print(f"\nTraining iteration {i+1}/5")
            
            # Create metric callback
            metric_callback = MetricCallback(role=training_role)
            
            if training_role == 'red':
                # Curriculum learning for defensive training
                curriculum_timesteps = steps_per_opponent // 3
                for difficulty in ['easy', 'medium', 'hard']:
                    print(f"\nTraining against {difficulty} opponent...")
                    self._adjust_opponent_difficulty(opponent, difficulty)
                    training_agent.learn(
                        total_timesteps=curriculum_timesteps,
                        reset_num_timesteps=False,
                        callback=metric_callback
                    )
            else:
                # Regular training for blue
                training_agent.learn(
                    total_timesteps=steps_per_opponent,
                    reset_num_timesteps=False,
                    callback=metric_callback
                )
            
            # Get and print metrics
            metrics = metric_callback.get_episode_statistics()
            print(f"\nIteration {i+1} Statistics for {training_role.upper()}:")
            print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"Average Reward: {metrics['avg_reward']:.2f}")
            print(f"Average Episode Length: {metrics['avg_length']:.1f}")
            
            # Evaluate current progress
            if i % 2 == 1:  # Every other iteration
                print("\nEvaluating current agents:")
                self.evaluate_agents(num_episodes=5)
        
        return metrics

    def train(self, num_phases=10, timesteps_per_phase=100000):
        """Full training loop alternating between agents"""
        try:
            phase_metrics = []
            for phase in range(num_phases):
                print(f"\n{'='*50}")
                print(f"Phase {phase + 1}/{num_phases}")
                print(f"{'='*50}")
                
                # Train blue (attacker)
                blue_metrics = self.train_phase('blue', timesteps_per_phase)
                self.blue_agent.save(f"{self.save_dir}/blue_latest")
                self.add_clone(self.blue_agent, self.blue_clones, phase + 1, 'blue')
                
                # Train red (defender)
                red_metrics = self.train_phase('red', timesteps_per_phase)
                self.red_agent.save(f"{self.save_dir}/red_latest")
                self.add_clone(self.red_agent, self.red_clones, phase + 1, 'red')
                
                # Store and print phase metrics
                phase_metrics.append({
                    'phase': phase + 1,
                    'blue': blue_metrics,
                    'red': red_metrics
                })
                
                self._print_phase_summary(phase_metrics)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving models...")
            self.blue_agent.save(f"{self.save_dir}/blue_interrupted")
            self.red_agent.save(f"{self.save_dir}/red_interrupted")
            print("Models saved. You can resume later.")

    def _print_phase_summary(self, phase_metrics):
        """Print summary of training phase results"""
        latest = phase_metrics[-1]
        print("\nPhase Performance Summary:")
        print(f"Blue - Win Rate: {latest['blue']['win_rate']*100:.1f}%, " 
              f"Avg Reward: {latest['blue']['avg_reward']:.2f}")
        print(f"Red  - Win Rate: {latest['red']['win_rate']*100:.1f}%, "
              f"Avg Reward: {latest['red']['avg_reward']:.2f}")
        
        if len(phase_metrics) >= 3:
            recent_blue = [m['blue']['win_rate'] for m in phase_metrics[-3:]]
            recent_red = [m['red']['win_rate'] for m in phase_metrics[-3:]]
            print("\nTrend (last 3 phases):")
            print(f"Blue: {recent_blue[0]*100:.1f}% → {recent_blue[-1]*100:.1f}%")
            print(f"Red:  {recent_red[0]*100:.1f}% → {recent_red[-1]*100:.1f}%")

    def evaluate_agents(self, num_episodes=10):
        """Run multiple evaluation matches and collect statistics"""
        print("\nRunning evaluation matches...")
        blue_wins = 0
        red_wins = 0
        
        env = DummyVecEnv([lambda: SwerveGameEnv(render=False)])
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            env.envs[0].set_role('blue')
            state = env.reset()
            done = [False]
            episode_steps = 0
            
            while not done[0]:
                # Blue's turn
                env.envs[0].set_role('blue')
                blue_action, _ = self.blue_agent.predict(state, deterministic=True)
                next_state, blue_reward, done, info = env.step([blue_action])
                
                if done[0]:
                    blue_top = env.envs[0].robot_blue.body.position.y * PPM - (env.envs[0].robot_blue.size * PPM / 2)
                    timeout = episode_steps >= GAME_DURATION * 60
                    blue_scored = blue_top < FIELD_MARGIN
                    
                    if blue_scored:
                        print("Blue scored!")
                        blue_wins += 1
                    elif timeout:
                        print("Red won by timeout!")
                        red_wins += 1
                    break
                
                # Red's turn
                env.envs[0].set_role('red')
                red_action, _ = self.red_agent.predict(next_state, deterministic=True)
                next_state, red_reward, done, info = env.step([red_action])
                
                if done[0]:
                    blue_top = env.envs[0].robot_blue.body.position.y * PPM - (env.envs[0].robot_blue.size * PPM / 2)
                    timeout = episode_steps >= GAME_DURATION * 60
                    blue_scored = blue_top < FIELD_MARGIN
                    
                    if blue_scored:
                        print("Blue scored!")
                        blue_wins += 1
                    elif timeout:
                        print("Red won by timeout!")
                        red_wins += 1
                    break
                
                state = next_state
                episode_steps += 2
        
        print("\nEvaluation Results:")
        print(f"Blue Wins: {blue_wins}/{num_episodes} ({blue_wins/num_episodes*100:.1f}%)")
        print(f"Red Wins: {red_wins}/{num_episodes} ({red_wins/num_episodes*100:.1f}%)")
        
        return blue_wins, red_wins

    def visualize_match(self, blue_agent, red_agent):
        """Visualize a single match between agents"""
        env = DummyVecEnv([lambda: SwerveGameEnv(render=True)])
        env.envs[0].set_role('blue')
        state = env.reset()
        done = [False]
        
        while not done[0]:
            # Blue's turn
            env.envs[0].set_role('blue')
            blue_action, _ = blue_agent.predict(state, deterministic=True)
            next_state, blue_reward, done, info = env.step([blue_action])
            
            if done[0]:
                break
            
            # Red's turn
            env.envs[0].set_role('red')
            red_action, _ = red_agent.predict(next_state, deterministic=True)
            next_state, red_reward, done, info = env.step([red_action])
            
            state = next_state
        
        env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phases', type=int, default=50,
                      help='Number of training phases')
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Timesteps per phase')
    args = parser.parse_args()
    
    trainer = SelfPlayPPOTrainer()
    trainer.train(num_phases=args.phases, timesteps_per_phase=args.timesteps)
