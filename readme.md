# DOGE: Distance-sensitive Offline rl with better GEneralization
DOGE (https://arxiv.org/abs/2205.11027) is an offline RL method designed from the perspective of generalization performance of deep function approximators. DOGE trains a state-conditioned distance function that can be readily plugged into standard actor-critic methods as a policy constraint. Simple yet elegant, our algorithm enjoys better generalization compared to state-of-the-art methods on D4RL benchmarks.

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
@article{li2022distance,
  title={Distance-Sensitive Offline Reinforcement Learning},
  author={Li, Jianxiong and Zhan, Xianyuan and Xu, Haoran and Zhu, Xiangyu and Liu, Jingjing and Zhang, Ya-Qin},
  journal={arXiv preprint arXiv:2205.11027},
  year={2022}
}
```

