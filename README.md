# **Reinforcement Learning Framework for Radical Optimization**

This repository provides a complete implementation of a reinforcement learning (RL) framework for ranking and optimizing radical candidates. The pipeline is designed to train lightweight neural agents that guide radical selection under descriptor constraints.  

The repository includes both modular **Python codes** and interactive **Jupyter notebooks** for reproducibility, benchmarking, and downstream analysis.

---

## **Repository Structure**

The project is organized into two main folders:

- **codes/** → Contains all Python modules and training scripts.  
- **notebook/** → Contains Jupyter notebooks for interactive experimentation and visualization.  

Additional folder:

- **data/** → Input CSV files (training, validation, benchmark).  

---

## **Methodology**

The reinforcement learning workflow implemented here follows these main stages:

### **1. Environment Setup**
- Candidate radicals are represented by tabular features.  
- An RL environment (`RadicalEnv`) provides `reset()` and `step()` for training interactions.  
- Files:  
  - `env.py` – environment logic  
  - `models.py` – DQN and DeltaNet network definitions
 
  ## **Usage**

Run training directly from the command line:
 
  python codess/scripts/run_training.py \
  --base_dir ./dataset \
  --episodes 500 \
  --lr 1e-4 \
  --hidden_size 128 \
  --num_hidden_layers 2 \
  --gamma 0.99 \
  --eps_decay 0.995 \
  --device cpu


### **2. Model Training**
- A Deep Q-Network (DQN) selects actions from state features.  
- A DeltaNet predicts small adjustments to descriptors.  
- Training uses epsilon-greedy exploration with standard RL updates.  
- Files:  
  - `train.py` – training loops and grid search  
  - `run_training.py` – command-line entry point with argparse  

### **3. Interactive Exploration**
- A Jupyter notebook provides a step-by-step demo.  
- Users can visualize intermediate states, inspect candidates, and generate figures interactively.  
- Files:  
  - `demo_notebook.ipynb` in the **notebook/** folder  

---

## **Installation**

Clone the repository and install dependencies:

    git clone https://github.com/Debojyoti91/RL_Radical_Optimization.git
    cd RL_Radical_Optimization
    pip install -r requirements.txt

---

This will load your training and validation data from **data/**, train an RL agent with the given hyperparameters, and save trained models and logs inside **codes/** (or paths you define in the script).  

For an interactive demo instead of CLI, launch:

    jupyter lab
    # then open notebook/RL_radical_optimization.ipynb

---

## **License**

This project is released under the MIT License. See LICENSE for details.

---

## **Citation**

If you use this repository, please cite:



