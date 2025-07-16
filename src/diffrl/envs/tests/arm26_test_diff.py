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

def step_fn(m, data, action):
    data = data.replace(ctrl=action)
    data = mjx.step(m, data)
    data = mjx.kinematics(m, data)
    return data

def rollout(m, data_init, action, steps=10):
    def iteration(carry, time):
        data = carry
        data = step_fn(m, data, action)
        return data, data
    time = jnp.arange(steps)
    data_final, trajectory = jax.lax.scan(
            iteration,
            mjx_data,
            time,
    )
    return data_final

def loss(action):
    final = rollout(mjx_model, mjx_data, action)
    return jnp.linalg.norm(final.qpos)
            
loss_grad = jax.jit(jax.grad(loss))
print(loss_grad(jnp.ones(mj_model.nu)))
