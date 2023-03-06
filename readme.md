# DOGE: When Data Geometry Meets Deep Function: Generalizing Offline Reinforcement Learning
DOGE (https://openreview.net/forum?id=lMO7TC7cuuh) is an offline RL method designed from the perspective of generalization performance of deep function approximators. DOGE trains a state-conditioned distance function that can be readily plugged into standard actor-critic methods as a policy constraint. Simple yet elegant, our algorithm enjoys better generalization compared to state-of-the-art methods on D4RL benchmarks.

#### Usage
To install the dependencies, use 
```python
    pip install -r requirements.txt
```
#### Benchmark experiments
You can run Mujoco tasks and AntMaze tasks like so:
```python
    python train_distance_mujoco.py --env_name halfcheetah-medium-v2 --alpha 7.5
```
```python
    python train_distance_antmaze.py --env_name antmaze-umaze-v2 --alpha 5.0
```
#### Modified AntMaze tasks

You can run the modified AntMaze medium/large tasks like so:
```python
    python train_distance_antmaze.py --env_name antmaze-large-play-v2 --alpha 70 --toycase True
```

#### Visulization of Learning curves
You can resort to [wandb](https://wandb.ai/site) to login your personal account via export your own wandb api key.
```
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
and run 
```
wandb online
```
to turn on the online syncronization.

#### Bibtex

```
@inproceedings{
li2023when,
title={When Data Geometry Meets Deep Function: Generalizing Offline Reinforcement Learning},
author={Jianxiong Li and Xianyuan Zhan and Haoran Xu and Xiangyu Zhu and Jingjing Liu and Ya-Qin Zhang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=lMO7TC7cuuh}
}
```

