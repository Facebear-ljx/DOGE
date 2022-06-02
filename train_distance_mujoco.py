import wandb
import argparse
from RL_algos.doge import DOGE
import datetime
import random
import os


def main():
    wandb.init(project="DOGE_mujoco")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='Solve Mujoco with DOGE')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='halfcheetah-medium-v2', help='choose your mujoco env')
    parser.add_argument('--alpha', default=7.5, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--negative_samples', default=20, type=int)
    parser.add_argument('--negative_policy', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--distance_steps', default=int(1e+5), type=int, help='total steps to train Distance function')
    parser.add_argument('--strong_contrastive', default=False)
    parser.add_argument('--scale_state', default='standard')
    parser.add_argument('--scale_action', default=False)
    parser.add_argument('--lr_distance', default=1e-3, type=float)
    parser.add_argument('--initial_lmbda', default=5., type=float)
    parser.add_argument('--lr_actor', default=3e-4, type=float)
    parser.add_argument('--lr_critic', default=3e-4, type=float)
    parser.add_argument('--lmbda_min', default=1, type=float)
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and DOGE agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{args.alpha}_{env_name}"

    agent_DOGE = DOGE(env_name=env_name,
                      device=args.device,
                      ratio=1,
                      seed=args.seed,
                      alpha=args.alpha,
                      negative_samples=args.negative_samples,
                      batch_size=args.batch_size,
                      distance_steps=args.distance_steps,
                      negative_policy=args.negative_policy,
                      strong_contrastive=args.strong_contrastive,
                      lmbda_min=args.lmbda_min,
                      scale_state=args.scale_state,
                      scale_action=args.scale_action,
                      lr_distance=args.lr_distance,
                      lr_actor=args.lr_actor,
                      lr_critic=args.lr_critic,
                      initial_lmbda=args.initial_lmbda,
                      gamma=args.gamma,
                      evalute_episodes=10
                      )

    agent_DOGE.learn(total_time_step=int(1e+6))


if __name__ == '__main__':
    main()
