import os
# os.environ['MUJOCO_GL'] = 'egl'
# os.environ['EQX_ON_ERROR'] = 'nan'
# os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'
# os.environ['JAX_ENABLE_X64'] = 'true'
# os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'high'
# os.environ['JAX_DEBUG_NANS'] = 'false'
# os.environ['JAX_COMPILATION_CACHE_DIR'] = os.environ['JAX_CACHE_DIR']
# os.environ['JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES'] = 'xla_gpu_per_fusion_autotune_cache_dir'

## Imports 
# basic libraries
from copy import Error
from pathlib import Path
from typing import Dict, List, Optional
import jax
import jax.numpy as jp

# brax imports
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# mujoco stuff
import mujoco
from mujoco import mjx

XML_PATH = (Path(__file__).parents[0] / 
            "data/arm26.xml")
DIFF_MJX_OVERWRITE_DICT = {
    ## Option (same as in)
    # CFD (enabled.yaml)
    'opt.cfd_enable': True,
    'opt.cfd_solimp': [0.0, 0.01, 0.05, 1.0, 2.0],
    # Colision (soft.yaml)
    'opt.col_soft_enable': True,
    # Integrator (diffrax.yaml)
    'opt.integrator': 'Diffrax',
    'opt.dfx_approximate_integration': False,
    'opt.dfx_max_ode_steps': 256, 
    'opt.dfx_solver': 'Tsit5',
    'opt.dfx_stepsize_controller': 'PID',
    'opt.dfx_pid_atol': 1e-2, # sehr hoch
    'opt.dfx_pid_rtol': 1e-2,
    'opt.dfx_adjoint': 'RecursiveCheckpoint',
    'opt.dfx_recursive_ncheckpoints': None, 
    'opt.dfx_pid_dt0': None,
    # Misc (myoarm_bionic_pingpong_2.yaml)
    'opt.timestep': 0.0025, 
    'opt.iterations': 4, # Das kann auf 1
    'opt.ls_iterations': 16
}

# helper function
def set_contact_properties(
    body: mujoco.MjModel, # type: ignore
    body_name: Optional[str] = None,
    not_body_name: Optional[str] = None,
    geom_type_name: Optional[str] = None,
    not_geom_type_name: Optional[str] = None,
    solref: Optional[List[float]] = None,
    solimp: Optional[List[float]] = None,
    contype: Optional[int] = None,
    conaffinity: Optional[int] = None
):
    if (
        (body_name is None or body_name in body.name) and
        (not_body_name is None or not (not_body_name in body.name))
    ):
        for geom in body.geoms:
            if (
                    (geom_type_name is None or 
                     geom_type_name in geom.type.name) and 
                    (not_geom_type_name is None or 
                     not (not_geom_type_name in geom.type.name))
            ):
                if solref is not None:
                    geom.solref = jp.array(solref)
                if solimp is not None:
                    geom.solimp = jp.array(solimp)
                if contype is not None:
                    geom.contype = jp.array(contype)
                if conaffinity is not None:
                    geom.conaffinity = jp.array(conaffinity)
    for child in body.bodies:
        set_contact_properties(
            child,
            body_name,
            not_body_name,
            geom_type_name,
            not_geom_type_name,
            solref,
            solimp,
            contype,
            conaffinity
        )


class Arm26(PipelineEnv):
    """
    Diff MJX arm26 environment
    """

    def __init__(
            self,
            reward_weights : Dict[str, float]  = {
                "position": 1.0,
                "effort": 0.15
            },  
            **kwargs
    ):
        # Load model
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

        # End effector site id
        self.end_effector_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_SITE.value,
            "end_effector"
        )

        # Make model an mjx system
        sys = mjcf.load_model(mj_model)
        # sys = mjx.put_model(mj_model)


        # Weird fix
        if sys.mesh_convex == ():
            sys = sys.replace(mesh_convex=None)

        # Add overwrite dict
        sys = sys.tree_replace(DIFF_MJX_OVERWRITE_DICT)

        # Save reward dict
        self.reward_weights = reward_weights

        # Call PipelineEnv __init__
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        qpos = self.sys.qpos0 
        qvel = jp.zeros_like(qpos)
        data = self.pipeline_init(qpos, qvel)
        metrics = {}
        info = {
            "goal": self._sample_goal(rng),
            "rewards": {
                "position": 0,
                "effort": 0,
            }
        }
        obs = self._get_obs(data, info["goal"])
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        current_data = state.pipeline_state
        updated_data = self.pipeline_step(current_data, action)
        rewards = self._get_rewards(updated_data, state.info["goal"])

        # Total reward as weighted sum
        reward = sum(
            self.reward_weights[sub_reward]
          * rewards[sub_reward]
            for sub_reward 
            in self.reward_weights.keys()
        )

        # Update state
        obs = self._get_obs(updated_data, state.info["goal"])
        done = 0.0
        state.info["rewards"] = rewards

        return state.replace( # type: ignore
            pipeline_state=updated_data, 
            obs=obs, 
            reward=reward, 
            done=done
        )

    def _get_rewards(
            self,
            data: mjx.Data,
            goal: jp.ndarray
    ) -> Dict["str", jp.ndarray]:
        # Get variables
        id = self.end_effector_id
        end_effector_xpos = data.site_xpos[id]
        activation = data.act

        # Calculate the rewards
        rewards = {
            "position": # Gaussian position reward
                jp.exp(
                  - jp.sum(  
                        (goal 
                       - end_effector_xpos[:2]) ** 2
                    ) / (0.2 ** 2)
                ),
            "effort":   # Mean square activation
                1 - jp.mean(activation ** 2)
        }

        return rewards

    def _sample_goal(self, rng: jp.ndarray) -> jp.ndarray:
        # Sample goal uniformly
        return jax.random.uniform(
            rng,
            shape=2,
            minval=-1.0,
            maxval=1.0
        )
        
    def _get_obs(
        self, 
        data: mjx.Data, 
        goal: jp.ndarray,
    ) -> jp.ndarray:

        return jp.concatenate([
            data.qpos,
            data.qvel,
            goal,
        ])
    
envs.register_environment('arm26', Arm26)
