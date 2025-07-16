## Imports
# Basic imports
import numpy as np
import jax
import jax.numpy as jp

# Environment Imports
import diffrl.envs.arm26
from brax import envs
from brax.io import html

## Rollout
# instantiate the environment
env_name = 'arm26'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
key = jax.random.key(666)
total = 1000
from tqdm import tqdm
for i in tqdm(range(total), total=total):
  key, _ = jax.random.split(key)
  ctrl = jax.random.uniform(key, env.sys.nu) # type: ignore
  ctrl = ctrl.at[0].set(0) 
  ctrl = ctrl.at[1].set(0) 
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

## Plot trajectory with HTML
html.save("html.html", env.sys, rollout)
