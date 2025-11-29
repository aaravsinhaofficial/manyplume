import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class PlumeEnvironment:
    def __init__(self, puff_df, wind_df, max_steps=1500, threshold=0.5, 
                 action_type='continuous', dt=0.01, source_location=(0,0)):
        self.puff_df = puff_df
        self.wind_df = wind_df
        self.max_steps = max_steps
        self.threshold = threshold
        self.action_type = action_type
        self.dt = dt
        self.source_location = source_location
        
        # Time management
        self.times = sorted(puff_df['time'].unique())
        self.current_time_idx = 0
        self.current_time = self.times[self.current_time_idx]
        
        # Agent state
        self.current_position = np.array([0.0, 0.0])  # Start at origin
        self.done = False
        
        # Setup action space
        self.action_space = type('', (), {})()
        self.action_space.sample = lambda: np.random.uniform(-1, 1, size=2)
        
        # Setup observation space (wind_x, wind_y, odor)
        self.observation_space = type('', (), {})()
        self.observation_space.shape = (3,)
    
    def reset(self):
        self.current_position = np.array([0.0, 0.0])
        self.current_time_idx = 0
        self.current_time = self.times[self.current_time_idx]
        self.done = False
        return self._get_observation()
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Update position (simple movement model)
        step_size, angle = action[0]  # Unpack action
        angle_rad = angle * 2 * np.pi  # Convert to radians
        
        self.current_position[0] += step_size * np.cos(angle_rad) * self.dt
        self.current_position[1] += step_size * np.sin(angle_rad) * self.dt
        
        # Advance time
        self.current_time_idx += 1
        if self.current_time_idx >= len(self.times):
            self.done = True
            return self._get_observation(), 0, True, {}
            
        self.current_time = self.times[self.current_time_idx]
        
        # Calculate reward (simple distance-based)
        distance = np.linalg.norm(self.current_position - self.source_location)
        reward = 1 / (1 + distance)
        
        return self._get_observation(), reward, self.done, {}
    
    def _get_observation(self):
        """Returns [wind_x, wind_y, odor] at current position"""
        wind_row = self.wind_df[self.wind_df['time'] == self.current_time].iloc[0]
        wind_x = wind_row['wind_x']
        wind_y = wind_row['wind_y']
        odor = self._get_odor_at_position(self.current_position)
        
        return np.array([wind_x, wind_y, odor])
    
    def _get_odor_at_position(self, position):
        current_puffs = self.puff_df[self.puff_df['time'] == self.current_time]
        if len(current_puffs) == 0:
            return 0.0
        
        positions = current_puffs[['x', 'y']].values
        distances = cdist([position], positions)[0]
        radii = current_puffs['radius'].values
        concentrations = 1 / (1 + (distances / (radii + 1e-5))**2)
        return np.sum(concentrations)