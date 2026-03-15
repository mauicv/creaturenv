## Parameterized Multi-Legged Thruster Creature (Gymnasium + Box2D)

2D zero-gravity creature navigation environment with:
- configurable leg topology (for example `[2, 1, 3]`)
- per-leg tip thrusters
- joint motor control
- lidar obstacle sensing
- target navigation objective

The purpose of this RL environment is to explore how world model training changes with increasing complexity of environment. In particular, the various parameters that you can set, such as the leg topology or the number of obsticals allows you to vary different aspects of the complexity in a smoother manner than switching from pendulum to bipedal walker.

## AI-Generated Code Notice

This project was initially generated with AI assistance and then iterated in-editor.
Please review behavior, safety constraints, and physics assumptions before using it for research or production work.

## Project Structure

- `creature_env/__init__.py` - env registration (`CreatureNavigation-v0`)
- `creature_env/creature_env.py` - main Gymnasium env
- `creature_env/creature_builder.py` - Box2D world and creature construction
- `creature_env/lidar.py` - lidar raycast callback/helpers
- `creature_env/renderer.py` - pygame renderer (`human` and `rgb_array`)
- `creature_env/test_env.py` - sanity test + visual run
- `examples/train_ppo.py` - Stable-Baselines3 PPO training entrypoint
- `examples/play_ppo.py` - run a trained PPO policy with rendering
- `requirements/base.txt` - Python dependencies
- `requirements/rl.txt` - RL training dependencies (SB3 + extras)

## Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements/base.txt
```

Notes:
- Current requirements use `Box2D>=2.3.10` (wheel-friendly on most setups).
- If you switch back to `box2d-py`, you may need system `swig`.

## Run the visual sanity check

From repo root:

```bash
python -m creature_env.test_env
```

If `python` is not available in your shell:

```bash
python3 -m creature_env.test_env
```

This opens a pygame window, runs random actions, renders each step, and prints `Sanity check passed.` when complete.

## Train with PPO (Stable-Baselines3)

Install RL dependencies:

```bash
pip install -r requirements/rl.txt
```

Train:

```bash
python -m examples.train_ppo --total-timesteps 500000 --run-name ppo_creature
```

Play a trained policy:

```bash
python -m examples.play_ppo --model-path runs/ppo_creature/checkpoints/best_model.zip --deterministic
```

## Quick Usage Example

```python
import gymnasium as gym
import creature_env  # registers CreatureNavigation-v0

env = gym.make("CreatureNavigation-v0", leg_spec=[2, 1, 3], render_mode="human")
obs, info = env.reset(seed=0)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

# PPO training script:

```sh
python -m examples.train_ppo \
  --run-name train_fast_laptop \
  --total-timesteps 30000 \
  --n-envs 2 \
  --max-episode-steps 300 \
  --leg-spec 1 \
  --num-obstacles 0 \
  --n-lidar-rays 4 \
  --lidar-range 6.0 \
  --fluid-friction 0.8 \
  --max-thrust 6.0 \
  --max-joint-torque 20.0 \
  --n-steps 256 \
  --batch-size 64
```