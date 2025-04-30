# ELE392 Reinforcement Learning with Fetch Robotics

> A comprehensive repository for training, evaluating, and demoing SAC-based reinforcement learning models on Gymnasium Robotics Fetch environments. Developed for ELE392 at the University of Rhode Island under Dr. Chiovaro.

---

## 🚀 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Folder Structure](#folder-structure)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
5. [Usage](#usage)  
   - [Training New Models](#training-new-models)  
   - [Continuing Training](#continuing-training)  
   - [Viewing Trained Models](#viewing-trained-models)  
   - [Gradio App Demo](#gradio-app-demo)  
6. [Code Specifics](#code-specifics)  
   - [create_env.py](#create_envpy)  
   - [Training Pipelines](#training-pipelines)  
7. [Reference](#reference)  
8. [License](#license)  

---

## 📝 Overview

This repository was created to manage the development of our group’s reinforcement learning project for **ELE392** at the University of Rhode Island, taught by Dr. Chiovaro. We implement and train Soft Actor-Critic (SAC) models on Gymnasium’s Robotics Fetch environments, achieving successful training on:

- **FetchPickAndPlace-v3**  
- **FetchReach-v3**  
- **FetchPush-v3**  

The Fetch environments are based on the 7-DoF Fetch Mobile Manipulator arm, with a two-fingered parallel gripper attached to it.

---

## ✨ Features

- 🔄 **Modular codebase** for separate tasks: pushing, reaching, picking & placing, and lifting.  
- 📈 **Weights & Biases** integration for logging metrics, video recordings, and model checkpoints.  
- 🤖 **Stable-Baselines3 SAC** implementations with Hindsight Experience Replay.  
- 🌐 **Gradio app** to demo trained agents interactively.  
- 🧪 **Custom environment wrapper** to fix joint-initialization bugs.  

---

## 📂 Folder Structure

```
├── App/                 # Gradio demo application  
├── push/                # Push environment code  
├── reach/               # Reach environment code  
├── lift/                # Custom PickAndPlace wrapper (lifting)  
├── Models/              # Saved trained model checkpoints  
├── old_stuff/           # Early prototype code  
├── create_env.py        # Env helper & bug-fix wrapper  
├── train_model.py       # Train a new SAC model from scratch  
├── train_upload.py      # Load & continue training an existing model  
├── view_model.py        # Render & visualize a trained agent  
└── README.md            # (you are here!)  
```

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.11.6 Recommended  
- `gymnasium`, `gymnasium_robotics`  
- `stable-baselines3`  
- `wandb`  
- `numpy`, `torch`, `Pillow`  

### Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/gkemp181/Reinforcement-Learning-Project.git
   cd Reinforcement-Learning-Project
   ```

2. Create & activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate       # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Training New Models

In ```train_model.py```, set these variables as desired.

```python
# Variables
run_name = "full_test3"
record_frequency = 10  # record the agent every n episodes
save_frequency = 100  # save the model every n episodes
show_progress_bar = True  # show progress bar during training
total_timesteps = 200_000  # total number of training timesteps
```

```bash
python train_model.py
```

- Initializes SAC, logs metrics & videos to W&B, and saves checkpoints periodically.

### Continuing Training

In ```train_upload.py```, set model path to model.zip file.

```python
# Path to model file
checkpoint_path = os.path.join("models", "recording_test2", "model.zip")
```

```bash
python train_upload.py
```

- Loads an existing `.zip` model, continues training with new timesteps, and logs under a new W&B run.

### Viewing Trained Models

```bash
python view_model.py
```

- Renders a trained policy in real time (human-viewable mode).

### Gradio App Demo

1. Navigate into the `App/` folder:

   ```bash
   cd App
   ```

2. Launch the demo:

   ```bash
   gradio app.py
   ```

3. Interactively select environment, load model, and watch the agent perform.

---

## 🔍 Code Specifics

### `create_env.py`

- **`create_env(render_mode=None)`**  
  - Registers Gymnasium Robotics environments.  
  - Wraps base Fetch environment with `RandomizedFetchWrapper`, which:
    - Fixes the robot-arm initial joint positions to sit above the table.  
    - Ensures object spawn location is never underneath the table surface.  

### Training Pipelines

Each training script (`train_model.py`, `train_upload.py`, etc.) follows this pattern:

1. **Environment initialization** via `create_env.py`.  
2. **Weights & Biases setup**:
   - Metric tracking (reward, loss, success rates).  
   - Periodic video recordings (`RecordVideo` wrapper).  
   - Automatic checkpoint saving every N steps.  
3. **SAC agent instantiation** with default hyperparameters.  
4. **Training loop** calling `model.learn(...)` with callbacks for logging & saving.

---

## 🤝 Reference

This project utilizes the Farama Foundation's Gymnasium Robotics API Fetch environments and the MuJoCo physics engine.

[Gymnasium Website](https://gymnasium.farama.org/)

[Gymnasium Robotics Documentation](https://robotics.farama.org/)

[Gymnasium Robotics Github](https://github.com/Farama-Foundation/Gymnasium-Robotics)

[MuJoCo Website](https://mujoco.org/)

```bibtex
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

> **ELE392 @ URI**  
> Dr. Chiovaro — Spring 2025
