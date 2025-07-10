## Imports 
# basic libraries
from copy import Error
from pathlib import Path
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
    'opt.integrator': 'Diffrax',
    'opt.dfx_approximate_integration': False,
    'opt.dfx_max_ode_steps': 256, 
    'opt.dfx_solver': 'Tsit5',
    'opt.dfx_stepsize_controller': 'PID',
    'opt.dfx_pid_atol': 0.01,
    'opt.dfx_pid_rtol': 0.01,
    'opt.dfx_adjoint': 'RecursiveCheckpoint',
    'opt.dfx_recursive_ncheckpoints': None, 
    'opt.dfx_pid_dt0': None,
    'opt.cfd_enable': True,
    'opt.cfd_solimp': [0.0, 0.01, 0.05, 1.0, 2.0],
    'opt.col_soft_enable': False,
    'opt.timestep': 0.0025, 
    'opt.iterations': 4,
    'opt.ls_iterations': 16
}

class Arm26(PipelineEnv):
    """
    TODO
    """

    def __init__(
            self,
            **kwargs
    ):
        # Load model
        mj_model = mujoco.MjModel.from_xml_path( # type: ignore
            XML_PATH.as_posix()
        )

        # Load model 
        sys = mjcf.load_model(mj_model)

        # Weird fix
        if sys.mesh_convex == ():
            sys = sys.replace(mesh_convex=None)

        # Add overwrite dict
        sys = sys.tree_replace(DIFF_MJX_OVERWRITE_DICT)
        
        # Put on device
        sys = jax.device_put(sys)

        # Call PipelineEnv __init__
        kwargs['backend'] = 'mjx'
        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:

        qpos = self.sys.qpos0 
        qvel = jp.zeros_like(qpos)
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.sys.nu))

        reward, done = jp.zeros(2)
        metrics = {}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        if data0 == None:
            raise Error("pipeline_state was none")

        data = self.pipeline_step(data0, action)

        obs = self._get_obs(data, action)
        reward = 0.0
        done = 0.0

        return state.replace( # type: ignore
            pipeline_state=data, 
            obs=obs, 
            reward=reward, 
            done=done
        )

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray
    ) -> jp.ndarray:

      # external_contact_forces are excluded
      return jp.concatenate([
          data.qpos,
          data.qvel,
      ])
    
envs.register_environment('arm26', Arm26)
