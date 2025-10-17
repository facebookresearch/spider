# Port new simulator

Start from exisitng RL environment, implement the following functions

First define env class to handle all states

Then implement the following functions

`setup_env`: prepare all env related resources

`step_env`: step the environment given the control action

`get_reward`: compute the reward for the current state

`get_terminal_reward`: compute the terminal reward for the current state, can be default to `get_reward`

`load_state` and `save_state`: load and save the environment state, important since in sampling, we need to constantly reset the environment

`save_env_params` and `load_env_params`: optional, mainly for curriculum. Can leave them empty

`sync_env`: for parallel env, need to broadcast the state from first env to all envs

Lastly, create a launcher
