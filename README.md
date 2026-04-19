# Tendon-Driven Dexterous Hand Library

A Python library for simulating, controlling, and training tendon-driven dexterous hands.

**This library does NOT include any robot model (URDF/STL).** You must provide your own URDF file if you want to run PyBullet simulation.

## Design Philosophy

- **Motor-centric API**: The library exposes motor commands, not joint targets.
- **Physics abstraction first**: Tendon routing, tension, and compliance are modeled in the core library.
- **Simulator-agnostic core**: PyBullet is an adapter, not a dependency of the physics core.
- **Composable architecture**: Finger, tendon, joint, actuator, hand, and controller are layered.
- **Deterministic by default**: Same input reproduces same output.

## Architecture

```text
User / Policy / RL
        ↓
Controller Layer
        ↓
Hand Model API
        ↓
Actuation Model  (motor → tendon displacement / tension)
        ↓
Tendon Physics Layer  (routing, friction, elasticity)
        ↓
Joint Response Layer  (torque, stiffness, limits, coupling)
        ↓
Simulation Adapter  (PyBullet / dummy backend / real hardware)
```

## Installation

### From source (development)

```bash
git clone git@github.com:JK5294/pybullet-simulation-tendow-hand-.git
cd pybullet-simulation-tendow-hand-
pip install -e ".[all]"
```

### From PyPI

```bash
pip install tendon-hand        # core only
pip install "tendon-hand[sim]" # core + PyBullet simulation
pip install "tendon-hand[rl]"  # core + RL dependencies
pip install "tendon-hand[all]" # everything
```

## Quick Start

### 1. Pure control (no simulator, no URDF needed)

```python
from tendon_hand.control.hand_controller import HandController

ctrl = HandController()

# Open / close poses
ctrl.open_pose()
ctrl.close_pose()

# Per-finger control
ctrl.set_index(m1=4.0, m2=3.0, m3=-0.3)

# Read joint targets
targets = ctrl.get_joint_targets()
print(targets["finger_2_link3_joint"])  # rad
```

### 2. Transmission model (motor → joint mapping, no URDF needed)

```python
from tendon_hand.core.models.transmission import CascadeTransmissionModel
import numpy as np

model = CascadeTransmissionModel()

# 17-D normalized motor action → 22-D joint targets
action = np.zeros(17)
action[3] = 1.0   # index_m1 full flex
action[4] = 0.8   # index_m2 partial flex
action[5] = 0.5   # index_m3 knuckle outward

targets = model.map(action)
```

### 3. PyBullet simulation (provide your own URDF)

```python
from tendon_hand.control.hand_controller import HandController
from tendon_hand.sim.pybullet_adapter import PyBulletHandAdapter

ctrl = HandController()

# You must provide your own URDF file
sim = PyBulletHandAdapter(
    hand=ctrl.hand,
    urdf_path="/path/to/your/hand.urdf",
    gui=True,
)
sim.reset()

# Close hand
ctrl.close_pose()
sim.apply_joint_targets(ctrl.get_joint_targets())
sim.step(120)

# Read actual joint angles
states = sim.get_joint_states()
pos, vel = states["finger_2_link3_joint"]
```

### 4. Gymnasium RL environment (provide your own URDF)

```python
import gymnasium as gym
import tendon_hand.env.gym_env  # registers the env

# You must provide your own URDF file
env = gym.make(
    "TendonHand-v1",
    render_mode="human",
    urdf_path="/path/to/your/hand.urdf",
)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Demos

### No URDF required — console demos with block visualization

These demos run entirely in the terminal and use an ASCII **block / tank** visualization to show how the under-actuated cascade works:

```bash
# Comprehensive library demo (control + block viz + transmission + inverse)
python examples/demo_hand.py all

# Step-by-step cable-driven cascade mechanics
python examples/demo_precise_angle.py index

# Finger-by-finger sign verification
python examples/demo_finger_test.py all
```

### Block Visualization Example

The demos show each motor as a "block" that cascades through joints like filling tanks:

```
Motor1 = 1.83 rad  [+███████░░░░░░░░░░░░░░░░░]
         └── drives: distal, middle

  Distal  (link3)  ████████████████████████   94.5°
  Middle  (link2)  ████░░░░░░░░░░░░░░░░░░░░   10.5°
  Proximal(link1)  █░░░░░░░░░░░░░░░░░░░░░░░    0.0°
```

### PyBullet demos (URDF required)

If you have a URDF, you can still run simulation-based demos:

```bash
python examples/demo_hand.py --urdf /path/to/your/hand.urdf sim
```

## Cable-Driven Cascade Mechanics

This is an **underactuated** tendon-driven hand. You cannot independently set each joint angle.

```
Motor1 (Tendon-1): distal → middle → proximal → palm
Motor2 (Tendon-2): middle (distal follows) → proximal → palm
Final angle      : max(motor1_effect, motor2_effect) per joint
```

**Key constraint**: To get proximal movement, distal and middle must already be at their limits.

See `examples/demo_precise_angle.py` for a step-by-step demonstration (no URDF required).

## Project Structure

```
tendon-hand/
├── src/tendon_hand/
│   ├── core/          # Physics-agnostic hand model
│   ├── control/       # High-level controller API
│   ├── sim/           # PyBullet adapter
│   ├── env/           # Gymnasium RL environment
│   ├── io/            # Config loading
│   └── utils/         # Math utilities + asset resolver
├── examples/          # Demo scripts (most do NOT require a URDF)
├── tests/             # pytest suite
└── pyproject.toml
```

**No robot models are included.** The library is designed to work with any tendon-driven hand URDF that follows the expected joint naming convention.

## Development

```bash
# Run tests (core tests do not require a URDF)
pytest tests/ -v

# Run simulation tests (provide your own URDF)
pytest tests/test_gym_env.py -v  # requires URDF in your local assets/

# Format code
black src/ tests/ examples/
ruff check src/ tests/ examples/
```

## License

MIT
