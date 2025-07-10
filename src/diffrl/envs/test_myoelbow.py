## Imports
# Basic imports
import numpy as np
import jax
import jax.numpy as jp

# Environment Imports
import diffrl.envs.myoelbow
from brax import envs

# Plotting imorts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('qt5agg')

## Rollout
# instantiate the environment
env_name = 'myoelbow'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
key = jax.random.key(666)
for i in range(1000):
  key, _ = jax.random.split(key)
  ctrl = jax.random.uniform(key, env.sys.nu) # type: ignore
  ctrl = ctrl.at[0].set(0) 
  ctrl = ctrl.at[1].set(0) 
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

## Plot trajectory with matplotlib
# Cast to np.arrays. Not sure if this is necessary.
frames = [
    np.array(frame, dtype=float) / 255 
    for frame in env.render(
        rollout,
        camera="top_view"
    ) # type: ignore
]
# Make axes
fig, ax = plt.subplots()
im = ax.imshow(frames[0])
plt.axis('off')

# Animation Update function
def update(frame):
    im.set_array(frame)
    return [im]

# Make animation
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames,
    interval=10,
    blit=True,
    repeat=False
)

# Show video
plt.show()
