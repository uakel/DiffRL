## Imports
# Basic imports
import numpy as np
import jax
import jax.numpy as jp

# Environment Imports
import mujoco
import diffrl.envs.arm26
from brax import envs

# Plotting imorts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('qt5agg')

## Rollout
# instantiate the environment
env_name = 'arm26'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

mj_model = env.sys.mj_model
data = mujoco.MjData(mj_model)

# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
trajectory = [state.pipeline_state]

# Make step function
def step_fn(state, action):
    state = jit_step(state, action)
    return state

action = jp.zeros(env.sys.nu)
action = action.at[0].set(1)
action = action.at[2].set(1)
action = action.at[4].set(1)

trajectory = []

from tqdm import tqdm
total=1000
for i in tqdm(range(total), total=total):
  state = step_fn(state, action)
  trajectory.append(state.pipeline_state)


def make_video(trajectory):
    renderer = mujoco.Renderer(mj_model)
    video = []
    for state in trajectory:
        data.qpos, data.qvel = state.qpos, state.qvel
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera="top_view")
        video.append(renderer.render())
    return video

def show_video(frames):
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
        interval=2.5,
        blit=True,
        repeat=False
    )
    
    # Show video
    ani.save('my_animation.mp4', writer='ffmpeg', fps=40)
    plt.show()

show_video(make_video(trajectory))
