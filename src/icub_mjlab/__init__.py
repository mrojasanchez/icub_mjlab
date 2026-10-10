from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import make_lift_cube_env_cfg
from .rl_cfg import liftcube_ppo_runner_cfg
from .rl_runner import IcubOnPolicyRunner

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-iCub",
  env_cfg=make_lift_cube_env_cfg(),
  play_env_cfg=make_lift_cube_env_cfg(play=True),
  rl_cfg=liftcube_ppo_runner_cfg(),
  runner_cls=IcubOnPolicyRunner,
)