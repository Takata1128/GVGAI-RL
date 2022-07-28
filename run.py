from algorithms.dqn import DQN
from config.config import DQNConfig
from wrapper.wrapper import make_gvgai_env

if __name__ == '__main__':
    config = DQNConfig()
    env = make_gvgai_env('gvgai-zelda-lvl0-v1')
    agent = DQN(env, config)
    agent.train()
