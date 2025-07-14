import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np
import pygame # type: ignore
import random
class SimpleSensorFusionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.width = 600
        self.height = 600
        self.ego_speed = 3
        self.dt = 0.1
        
        # Observation space: fused sensor data [distance, angle, type_confidence]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Action space: steering [-1 = left, 0 = straight, 1 = right]
        self.action_space = spaces.Discrete(3)
        
        # Graphics setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Simple Sensor Fusion")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)
        
        # Environment state
        self.ego_pos = None
        self.obstacles = []
        self.goal_pos = None
        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize ego vehicle at bottom center
        self.ego_pos = np.array([self.width//2, self.height-50])
        
        # Create random obstacles
        self.obstacles = []
        for _ in range(3):
            self.obstacles.append({
                'pos': np.array([random.randint(50, self.width-50), 
                                random.randint(100, self.height-100)]),
                'type': random.choice(['static', 'moving']),
                'radius': random.randint(15, 30)
            })
        
        # Set goal at top
        self.goal_pos = np.array([self.width//2, 30])
        
        self.current_step = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Simplified sensor fusion - returns closest obstacle info"""
        if not self.obstacles:
            return np.array([1.0, 0.5, 0.0])  # Default when no obstacles
        
        # Find closest obstacle
        closest = min(self.obstacles, 
         key=lambda o: np.linalg.norm(o['pos']-self.ego_pos))
        
        # Calculate relative position
        rel_pos = closest['pos'] - self.ego_pos
        distance = np.linalg.norm(rel_pos) / max(self.width, self.height)
        angle = np.arctan2(rel_pos[0], -rel_pos[1]) / np.pi  # Normalized [-1,1]
        
        # Type confidence (0=static, 1=moving)
        type_conf = 1.0 if closest['type'] == 'moving' else 0.0
        
        return np.array([distance, angle, type_conf])
    
    def step(self, action):
        # Move ego vehicle
        if action == 0:  # Left
            self.ego_pos[0] -= self.ego_speed * 2
        elif action == 1:  # Right
            self.ego_pos[0] += self.ego_speed * 2
        # Else straight (no x change)
        
        self.ego_pos[1] -= self.ego_speed  # Always move up
        
        # Move any moving obstacles
        for obs in self.obstacles:
            if obs['type'] == 'moving':
                obs['pos'][0] += random.uniform(-2, 2)
                obs['pos'][1] += random.uniform(-1, 1)
        
        # Check collisions
        terminated = False
        reward = 0.1  # Small step reward
        
        for obs in self.obstacles:
            if np.linalg.norm(obs['pos']-self.ego_pos) < (obs['radius'] + 20):
                terminated = True
                reward = -10
                break
        
        # Check if reached goal
        if np.linalg.norm(self.goal_pos-self.ego_pos) < 30:
            terminated = True
            reward = 10
        
        # Check max steps
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        self.screen.fill(self.white)
        
        # Draw goal
        pygame.draw.circle(self.screen, self.green, self.goal_pos, 15)
        
        # Draw obstacles
        for obs in self.obstacles:
            color = self.red if obs['type'] == 'static' else self.yellow
            pygame.draw.circle(self.screen, color, obs['pos'].astype(int), obs['radius'])
        
        # Draw ego vehicle
        pygame.draw.circle(self.screen, self.blue, self.ego_pos.astype(int), 20)
        
        # Draw sensor data
        obs_data = self._get_obs()
        text = f"Distance: {obs_data[0]:.2f} | Angle: {obs_data[1]:.2f} | Type: {'Moving' if obs_data[2] > 0.5 else 'Static'}"
        text_surface = self.font.render(text, True, self.black)
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(15)
    
    def close(self):
        pygame.quit()

# Example usage
if __name__ == "__main__":
    env = SimpleSensorFusionEnv()
    obs, _ = env.reset()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action = random.randint(0, 2)  # Random actions for demo
        obs, reward, terminated, truncated, _ = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close