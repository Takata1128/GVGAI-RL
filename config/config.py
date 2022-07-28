from dataclasses import dataclass


@dataclass
class Config:
    device: str = 'cuda'

    optimizer: str = 'Adam'

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 32

    model_dir: str = '/root/mnt/GVGAI-RL/checkpoints'


@dataclass
class DQNConfig(Config):
    env: str = 'zelda'
    id: str = 'v1'

    project: str = 'DDQN on Zelda Level1'

    n_steps: int = int(5e6)

    train_every: int = 50
    save_every: int = int(5e4)

    buffer_size: int = int(1e6)
    seed_buffer_size: int = int(1e4)

    target_update_interval: int = 2000

    beta_begin: float = 0.4
    beta_end: float = 1.0
    beta_decay: float = 5000000

    epsilon_begin: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 500000
