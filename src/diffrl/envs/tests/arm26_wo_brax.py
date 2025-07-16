## Imports
# Environment variables
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EQX_ON_ERROR'] = 'nan'
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'high'
os.environ['JAX_DEBUG_NANS'] = 'false'
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.environ['JAX_CACHE_DIR']
os.environ['JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES'] = 'xla_gpu_per_fusion_autotune_cache_dir'

# Basic imports
import numpy as np
import jax
import jax.numpy as jnp

# Environment Imports
import mujoco
from mujoco import mjx
from diffrl.envs.arm26 import (
    XML_PATH, 
    DIFF_MJX_OVERWRITE_DICT,
    set_contact_properties
)
from mujoco.mjx._src.io import _strip_weak_type

# Plotting imorts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('qt5agg')

# Upscale
def upscale(x):
    """Convert data to 64bit as make_data gives data in 32bit
    Code source: https://github.com/google-deepmind/mujoco/issues/2237"""
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

## Rollout
# instantiate the environment
specification = mujoco.MjSpec.from_file( # type: ignore
    XML_PATH.as_posix()
)

# Set contact properties
set_contact_properties(
    specification.worldbody,
    solimp=DIFF_MJX_OVERWRITE_DICT["opt.cfd_solimp"],
)
set_contact_properties(
    specification.worldbody,
    geom_type_name="MESH",
    contype=0,
    conaffinity=0
)

mj_model = specification.compile()
mjx_model = mjx.put_model(mj_model, _full_compat=True)

# Add overwrite dict
mjx_model = mjx_model.tree_replace(DIFF_MJX_OVERWRITE_DICT)
mjx_model = jax.device_put(mjx_model)
mjx_model = _strip_weak_type(mjx_model)
mj_model.opt.timestep = np.array(mjx_model.opt.timestep)

if mjx_model.mesh_convex == ():
    mjx_model = mjx_model.replace(mesh_convex=None)

data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, data, 0)

# initialize the state
mjx_data = mjx.put_data(mj_model, data)
mjx_data = mjx.kinematics(mjx_model, mjx_data)
mjx_data = jax.tree.map(upscale, mjx_data)

# Make step function
@jax.jit
def step_fn(m, data, action):
    data = data.replace(ctrl=action)
    data = mjx.step(m, data)
    data = mjx.kinematics(m, data)
    return data

action = jnp.zeros(mjx_model.nu)
action = action.at[0].set(1)
action = action.at[2].set(1)
action = action.at[4].set(1)

trajectory = []

from tqdm import tqdm
total=10
for i in tqdm(range(total), total=total):
    mjx_data = step_fn(mjx_model, mjx_data, action)
    trajectory.append(mjx_data)


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

ncon_over_time = jnp.array([d.ncon for d in trajectory])
print("Number of contacts each step:", ncon_over_time)
print("Max contacts at a step:", ncon_over_time.max())
print("Average contacts per step:", ncon_over_time.mean())

show_video(make_video(trajectory))
