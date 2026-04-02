# 1. ALWAYS IMPORT AI FIRST ON WINDOWS
import torch
from stable_baselines3 import PPO

# 2. THEN IMPORT THE 3D ENGINE
from ursina import *
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- 1. THE HARD MODE ENVIRONMENT ---
class CleanDrainEnv_HardMode(gym.Env):
    def __init__(self, grid_size=20, max_steps=400):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([grid_size-1, grid_size-1, grid_size-1, grid_size-1, grid_size-1, grid_size-1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.blockage_pos = np.random.randint(0, self.grid_size, size=2)
        self.bot_pos = np.random.randint(0, self.grid_size, size=2)
        while np.array_equal(self.bot_pos, self.blockage_pos):
            self.bot_pos = np.random.randint(0, self.grid_size, size=2)
        self.prev_pos = self.bot_pos.copy()
            
        self.obstacles = []
        # HARD MODE UPGRADE: 14 Walls, Max Length 8, L-Shape Corners
        num_walls = 14 
        max_wall_length = 8
        
        for _ in range(num_walls):
            start_x = np.random.randint(0, self.grid_size)
            start_y = np.random.randint(0, self.grid_size)
            is_vertical = np.random.choice([True, False])
            wall_length = np.random.randint(3, max_wall_length + 1)
            
            # 50% chance to create an L-Shape by bending the wall halfway
            bend_point = np.random.randint(2, wall_length) if np.random.rand() > 0.5 else 999
            
            curr_x, curr_y = start_x, start_y
            
            for i in range(wall_length):
                # Bend the wall
                if i == bend_point:
                    is_vertical = not is_vertical
                
                if is_vertical: curr_y = min(curr_y + 1, self.grid_size - 1)
                else: curr_x = min(curr_x + 1, self.grid_size - 1)
                
                obs_pos = np.array([curr_x, curr_y])
                
                if not (np.array_equal(obs_pos, self.bot_pos) or np.array_equal(obs_pos, self.blockage_pos)):
                    if not any(np.array_equal(obs_pos, o) for o in self.obstacles):
                        self.obstacles.append(obs_pos)
        
        return self._get_obs(), {}

    def _raycast(self, dx, dy):
        for step in range(1, 4):
            check_pos = np.array([self.bot_pos[0] + (dx * step), self.bot_pos[1] + (dy * step)])
            if check_pos[0] < 0 or check_pos[0] >= self.grid_size or check_pos[1] < 0 or check_pos[1] >= self.grid_size:
                if step == 1: return -1.0
                if step == 2: return 0.0
                return 0.5
            if any(np.array_equal(check_pos, o) for o in self.obstacles):
                if step == 1: return -1.0
                if step == 2: return 0.0
                return 0.5
        return 1.0

    def _get_obs(self):
        battery_level = 1.0 - (self.current_step / self.max_steps)
        return np.array([
            self.bot_pos[0], self.bot_pos[1], self.prev_pos[0], self.prev_pos[1], 
            self.blockage_pos[0], self.blockage_pos[1], battery_level,
            self._raycast(0, 1), self._raycast(0, -1), self._raycast(-1, 0), self._raycast(1, 0),
            self._raycast(1, 1), self._raycast(-1, 1), self._raycast(1, -1), self._raycast(-1, -1)
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        new_pos = self.bot_pos.copy()
        if action == 0:   new_pos[1] += 1 
        elif action == 1: new_pos[1] -= 1 
        elif action == 2: new_pos[0] -= 1 
        elif action == 3: new_pos[0] += 1 

        hit_obstacle = any(np.array_equal(new_pos, o) for o in self.obstacles)
        hit_wall = new_pos[0] < 0 or new_pos[0] >= self.grid_size or new_pos[1] < 0 or new_pos[1] >= self.grid_size

        if not (hit_obstacle or hit_wall):
            self.prev_pos = self.bot_pos.copy() 
            self.bot_pos = new_pos 

        terminated = np.array_equal(self.bot_pos, self.blockage_pos)
        truncated = self.current_step >= self.max_steps
        return self._get_obs(), 0, terminated, truncated, {}


# --- 2. ENGINE & LIGHTING SETUP ---
app = Ursina(borderless=False, multisampling=2)
window.title = 'CleanDrain 3D Showcase (Hard Mode)'

env = CleanDrainEnv_HardMode()
obs, _ = env.reset()

sun = DirectionalLight(shadows=True)
sun.position = (env.grid_size/2, 20, -10)
sun.look_at((env.grid_size/2, 0, env.grid_size/2))
Sky() 

try:
    model = PPO.load("best_model1.zip")
except:
    print("⚠️ WARNING: best_model1.zip not found!")
    model = None

ui_status = Text(text="Bot is running facility sweep...", position=(-0.85, 0.45), scale=2, color=color.white, background=True)
ui_instructions = Text(text="Press 'R' for New Maze", position=(-0.85, 0.40), scale=1.2, color=color.gray, background=True)

# --- 3. THE 2D MINIMAP HUD ---
minimap_bg = Entity(parent=camera.ui, model='quad', color=color.hex('#1a1a2e'), scale=(0.3, 0.3), position=(0.7, -0.35))
mm_cell = 1 / env.grid_size
mm_offset = -0.5 + (mm_cell / 2)

minimap_target = Entity(parent=minimap_bg, model='quad', color=color.red, scale=(mm_cell, mm_cell), z=-1)
minimap_bot = Entity(parent=minimap_bg, model='quad', color=color.cyan, scale=(mm_cell, mm_cell), z=-2)
mm_wall_entities = []

# --- 4. CREATE 3D ASSETS ---
ground = Entity(model='plane', scale=(env.grid_size, env.grid_size), texture='dots', color=color.hex('#0a0c10'), position=(env.grid_size/2 - 0.5, -0.5, env.grid_size/2 - 0.5))
bot_parent = Entity(position=(env.bot_pos[0], 0, env.bot_pos[1]))
robot_chassis = Entity(model='cylinder', parent=bot_parent, scale=(0.8, 0.4, 0.8), color=color.light_gray, origin_y=-0.5)
robot_turret = Entity(model='sphere', parent=robot_chassis, scale=(0.3, 0.3, 0.3), position=(0, 0.6, 0.3), color=color.lime)
target_base = Entity(position=(env.blockage_pos[0], 0, env.blockage_pos[1]))
target_core = Entity(model='sphere', color=color.red, parent=target_base, scale=1.3, alpha=0.8)

wall_entities = []
trail_entities = []

def build_visuals():
    global wall_entities, trail_entities, mm_wall_entities
    
    for w in wall_entities: destroy(w)
    for b in trail_entities: destroy(b)
    for mw in mm_wall_entities: destroy(mw)
    wall_entities.clear()
    trail_entities.clear()
    mm_wall_entities.clear()
    
    # 3D Walls and Minimap Walls
    for obs_pos in env.obstacles:
        w = Entity(model='cube', color=color.hex('#4a5568'), position=(obs_pos[0], 0, obs_pos[1]), scale=(1, 1.5, 1), collider='box', origin_y=-0.5)
        wall_entities.append(w)
        # Add to minimap
        mw = Entity(parent=minimap_bg, model='quad', color=color.gray, scale=(mm_cell, mm_cell), position=(mm_offset + obs_pos[0]*mm_cell, mm_offset + obs_pos[1]*mm_cell), z=-1)
        mm_wall_entities.append(mw)
        
    bot_parent.position = (env.bot_pos[0], 0, env.bot_pos[1])
    target_base.position = (env.blockage_pos[0], 0, env.blockage_pos[1])
    
    # Update Minimap Targets
    minimap_target.position = (mm_offset + env.blockage_pos[0]*mm_cell, mm_offset + env.blockage_pos[1]*mm_cell)

build_visuals()

# --- 5. POV CAMERA ---
camera.parent = bot_parent
camera.position = (0, 3.5, -6) # Pulled down and tight behind the bot
camera.look_at(bot_parent)
camera.rotation_x = 25 # Slight tilt down

# --- 6. SIMULATION LOGIC ---
simulation_running = True
step_timer = 0
trail_timer = 0

def update():
    global obs, step_timer, trail_timer, simulation_running

    if not simulation_running or model is None:
        return

    step_timer += time.dt
    trail_timer += time.dt
    
    # Breadcrumbs
    if trail_timer > 0.5:
        trail_timer = 0
        breadcrumb = Entity(model='sphere', position=bot_parent.position, scale=0.25, color=color.cyan, alpha=0.8, origin_y=-0.5)
        trail_entities.append(breadcrumb)
        for b in trail_entities:
            b.alpha -= 0.15 * time.dt
            b.scale -= 0.05 * time.dt
            if b.alpha < 0.1:
                trail_entities.remove(b)
                destroy(b)
                
    # AI Loop
    if step_timer > 0.15: # Faster movement for hard mode
        step_timer = 0
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action.item())
        
        bot_parent.animate_position((env.bot_pos[0], 0, env.bot_pos[1]), duration=0.1, curve=curve.linear)
        
        # Update Minimap Bot Position
        minimap_bot.position = (mm_offset + env.bot_pos[0]*mm_cell, mm_offset + env.bot_pos[1]*mm_cell)
        
        if terminated:
            ui_status.text = "✅ Facility Scan Complete."
            ui_status.color = color.lime
            simulation_running = False
        elif truncated:
            ui_status.text = "🔋 Low Battery. Trap Detected."
            ui_status.color = color.red
            simulation_running = False

# --- 7. RESTART ---
def input(key):
    global obs, simulation_running
    if key == 'r':
        obs, _ = env.reset()
        build_visuals()
        ui_status.text = "Bot is running facility sweep..."
        ui_status.color = color.white
        simulation_running = True

app.run()