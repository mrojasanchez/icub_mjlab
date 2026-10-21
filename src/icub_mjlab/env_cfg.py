"""iCub manipulation task environment configuration."""

import math
from pathlib import Path
import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.viewer import ViewerConfig
from mjlab.envs import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from icub_mjlab.icub_constants import (
  ICUB_ACTION_CLIP,
  ICUB_ACTION_SCALE,
  get_icub_robot_cfg,
)
from icub_mjlab.dextrah_rgb.actions.fabric_action import FabricActionCfg
import icub_mjlab.rewards as icub_rewards
import icub_mjlab.observations as icub_obs

_HERE = Path(__file__).parent
_TABLE_XML = _HERE / "table.xml"
_FLOOR_TEX = _HERE / "assets" / "textures" / "light-gray-floor-tile.png"
_WALLS_TEX = _HERE / "assets" / "textures" / "light-gray-plaster.png"


def get_table_spec() -> mujoco.MjSpec:
  """Load the table spec with collision and visual geoms."""
  return mujoco.MjSpec.from_file(str(_TABLE_XML))


def get_cube_spec(
  cube_size: float = 0.05,
  mass: float = 0.1,
  rgba: tuple[float, float, float, float] = (0.0, 0.2, 0.8, 1.0),
  with_stack_site: bool = False,
) -> mujoco.MjSpec:
  """Create a cube spec with physical and visual properties."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size / 2,) * 3,
    mass=mass,
    rgba=rgba,
    friction=(1.0, 0.005, 0.0001),
  )
  if with_stack_site:
    body.add_site(
      name="stack_target_top",
      pos=(0.0, 0.0, cube_size / 2 + 0.001),
      size=(0.002, 0.002, 0.002),
      rgba=(1.0, 0.0, 0.0, 0.0),
    )
  return spec


def add_room_visuals(spec: mujoco.MjSpec) -> None:
  """Add room walls, lighting, and camera settings matching the cube stacking scene."""
  spec.visual.map.zfar = 30.0
  spec.visual.quality.shadowsize = 2048
  spec.visual.rgba.haze = (0.15, 0.25, 0.35, 1.0)
  spec.visual.global_.offwidth = 800
  spec.visual.global_.offheight = 800

  spec.add_texture(
    name="tex-light-gray-plaster",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    file=str(_WALLS_TEX),
  )
  mat = spec.add_material(
    name="walls_mat",
    texrepeat=(3.0, 3.0),
    texuniform=True,
    specular=0.1,
    shininess=0.1,
    reflectance=0.0,
  )
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "tex-light-gray-plaster"

  # walls = [
  #   ("wall_leftcorner_visual", (1.25, -2.25, 1.5), (-0.2705981, -0.2705981, 0.6532815, 0.6532815), (1.06, 1.5, 0.01)),
  #   ("wall_rightcorner_visual", (1.25, 2.25, 1.5), (0.2705981, 0.2705981, 0.6532815, 0.6532815), (1.06, 1.5, 0.01)),
  #   ("wall_left_visual", (-1.25, -3.0, 1.5), (0.0, 0.0, 0.7071, 0.7071), (1.75, 1.5, 0.01)),
  #   ("wall_right_visual", (-1.25, 3.0, 1.5), (0.0, 0.0, 0.7071, -0.7071), (1.75, 1.5, 0.01)),
  #   ("wall_rear_visual", (2.0, 0.0, 1.5), (-0.5, -0.5, 0.5, 0.5), (1.5, 1.5, 0.01)),
  #   ("wall_front_visual", (-3.0, 0.0, 1.5), (0.5, 0.5, 0.5, 0.5), (3.0, 1.5, 0.01)),
  # ]
  # for name, pos, quat, size in walls:
  #   spec.worldbody.add_geom(
  #     name=name,
  #     type=mujoco.mjtGeom.mjGEOM_BOX,
  #     pos=pos,
  #     quat=quat,
  #     size=size,
  #     material="walls_mat",
  #     contype=0,
  #     conaffinity=0,
  #     group=1,
  #   )

  # spec.worldbody.add_camera(
  #   name="front_camera",
  #   pos=(-1.6, 0.0, 1.0),
  #   quat=(-0.56, -0.43, 0.43, 0.56),
  # )
  # spec.worldbody.add_camera(
  #   name="birdview",
  #   pos=(-0.5, 0.0, 2.0),
  #   quat=(-0.7071, 0.0, 0.0, 0.7071),
  # )
  # spec.worldbody.add_camera(
  #   name="agentview",
  #   pos=(-0.5, 0.0, 1.35),
  #   quat=(-0.653, -0.271, 0.271, 0.653),
  # )
  # spec.worldbody.add_camera(
  #   name="sideview",
  #   pos=(0.0565, -1.276, 1.488),
  #   quat=(-0.806, -0.591, 0.007, 0.010),
  # )


def make_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create base cube lifting task configuration.

  Args:
    play: If True, disables corruption and extends episode length for evaluation.
  """

  # ==============================================================================
  # Actions
  # ==============================================================================

  actions: dict[str, ActionTermCfg] = {
    "fabric": FabricActionCfg(
      entity_name="robot",
      site_name="r_hand_dh_frame_site",
      delta_pos_scale=0.05,
      delta_ori_scale=0.1,
    )
  }


  # ==============================================================================
  # Observations (DextrAH Asymmetric Actor-Critic)
  # ==============================================================================

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "hand_keypoints": ObservationTermCfg(
      func=icub_obs.hand_keypoints_pos,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "cube_pose": ObservationTermCfg(
      func=icub_obs.object_pose,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "goal_pos": ObservationTermCfg(
      func=icub_obs.target_goal_pos,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "fabric_state": ObservationTermCfg(
      func=icub_obs.fabric_state,
      params={"action_name": "fabric"},
    ),
  }

  critic_terms = {
    # Clean ground-truth versions of actor observations
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "hand_keypoints": ObservationTermCfg(func=icub_obs.hand_keypoints_pos),
    "cube_pose": ObservationTermCfg(func=icub_obs.object_pose),
    "goal_pos": ObservationTermCfg(func=icub_obs.target_goal_pos),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "fabric_state": ObservationTermCfg(
      func=icub_obs.fabric_state,
      params={"action_name": "fabric"},
    ),
    # Privileged critic-only observations
    "cube_velocity": ObservationTermCfg(func=icub_obs.object_velocity),
    "joint_torques": ObservationTermCfg(func=icub_obs.measured_joint_torques),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ==============================================================================
  # Rewards (DextrAH 4 terms + Arm Maintenance)
  # ==============================================================================

  rewards = {
    "hand_to_object": RewardTermCfg(
      func=icub_rewards.hand_to_object_dextrah_reward,
      weight=1.0,
      params={"scale": 10.0},
    ),
    "object_to_goal": RewardTermCfg(
      func=icub_rewards.object_to_goal_dextrah_reward,
      weight=2.0,
      params={"beta": 15.0, "min_lift_height": 0.670},
    ),
    "object_lift": RewardTermCfg(
      func=icub_rewards.object_lift_dextrah_reward,
      weight=1.5,
      params={"beta": 50.0},
    ),
    "finger_curl": RewardTermCfg(
      func=icub_rewards.finger_curl_regularization_reward,
      weight=1.0,
      params={"beta": 0.01},
    ),
    "left_arm_pose": RewardTermCfg(
      func=icub_rewards.frozen_arm_posture_reward,
      weight=0.05,
    ),
  }

  # ==============================================================================
  # Terminations
  # ==============================================================================

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "cube_dropped": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.50, "asset_cfg": SceneEntityCfg("cube")},
    ),
  }

  # ==============================================================================
  # Curriculum Learning (DextrAH-RGB Two-Stage Reward Curriculum)
  # ==============================================================================

  curriculum = {
    "hand_to_object_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "hand_to_object",
        "stages": [
          {"step": 0, "weight": 1.0},
          {"step": 24_000, "weight": 0.5},
        ],
      },
    ),
    "object_lift_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "object_lift",
        "stages": [
          {"step": 0, "weight": 1.5},
          {"step": 24_000, "weight": 2.0},
        ],
      },
    ),
    "object_to_goal_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "object_to_goal",
        "stages": [
          {"step": 0, "weight": 2.0},
          {"step": 24_000, "weight": 3.0},
        ],
      },
    ),
  }

  # ==============================================================================
  # Events & Commands
  # ==============================================================================

  commands: dict[str, CommandTermCfg] = {
    "lift_height": LiftingCommandCfg(
      entity_name="cube",
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
      object_pose_range=LiftingCommandCfg.ObjectPoseRangeCfg(
        x=(-0.40, -0.30),
        y=(0.05, 0.20),
        z=(0.625, 0.625),
        yaw=(-math.pi, math.pi),
      ),
      target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
        x=(-0.40, -0.30),
        y=(0.05, 0.20),
        z=(0.725, 0.825),
      ),
    )
  }

  events = {
    # Position robot base at env_origins
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Position table at env_origins
    "reset_table": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "asset_cfg": SceneEntityCfg("table"),
      },
    ),
    # Position red cube at env_origins
    "reset_red_cube": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("red_cube"),
      },
    ),
  }

  # Collision sensor for end-effector to ground/table contact.
  ee_ground_collision_cfg = ContactSensorCfg(
    name="ee_ground_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern=".*_hand",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern=".*", entity="table"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # ==============================================================================
  # Scene Configuration
  # ==============================================================================

  scene_cfg = SceneCfg(
    num_envs=64 if not play else 8,  # Fewer envs for play mode
    extent=2.0,   # Spacing between environments
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      textures=(
        spec_cfg.TextureCfg(
          name="arena_texplane",
          type="2d",
          file=str(_FLOOR_TEX),
        ),
      ),
      materials=(
        spec_cfg.MaterialCfg(
          name="floorplane",
          texture="arena_texplane",
          texrepeat=(2.0, 2.0),
          texuniform=True,
          reflectance=0.01,
          geom_names_expr=("terrain",),
        ),
      ),
      lights=(
        spec_cfg.LightCfg(
          name="spot_light",
          pos=(-1.0, -1.0, 1.5),
          dir=(0.2, 0.2, -1.0),
          specular=(0.3, 0.3, 0.3),
          type="directional",
          castshadow=False,
        ),
        spec_cfg.LightCfg(
          name="top_light",
          pos=(0.0, 0.0, 4.0),
          dir=(0.0, 0.0, -1.0),
          diffuse=(0.8, 0.8, 0.8),
          specular=(0.3, 0.3, 0.3),
        ),
      ),
    ),
    entities={
      "robot": get_icub_robot_cfg(),
      "table": EntityCfg(
        spec_fn=get_table_spec,
        init_state=EntityCfg.InitialStateCfg(
          pos=(-0.6, 0.0, 0.575),
          rot=(0.0, 0.0, 0.0, 1.0),
        ),
      ),
      "cube": EntityCfg(
        spec_fn=lambda: get_cube_spec(
          cube_size=0.05,
          mass=0.1,
          rgba=(0.0, 0.2, 0.8, 1.0),
        ),
        init_state=EntityCfg.InitialStateCfg(
          pos=(-0.35, 0.1, 0.625),
          rot=(0.0, 0.0, 0.0, 1.0),
        ),
      ),
      "red_cube": EntityCfg(
        spec_fn=lambda: get_cube_spec(
          cube_size=0.05,
          mass=0.1,
          rgba=(0.8, 0.1, 0.1, 1.0),
          with_stack_site=True,
        ),
        init_state=EntityCfg.InitialStateCfg(
          pos=(-0.35, -0.1, 0.625),
          rot=(0.0, 0.0, 0.0, 1.0),
        ),
      ),
    },
    sensors=(ee_ground_collision_cfg,),
    spec_fn=add_room_visuals,
  )

  viewer_cfg = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="chest",
    distance=9.0,
    elevation=-3.0,
    azimuth=-17.0,
  )

  sim_cfg = SimulationCfg(
    nconmax=128,
    njmax=600,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
    ),
  )

  # ==============================================================================
  # Environment Configuration
  # ==============================================================================

  return ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    sim=sim_cfg,
    viewer=viewer_cfg,
    decimation=4,           # 4 * 0.005s = 0.02s (50 Hz control)
    episode_length_s=int(1e9) if play else 10.0,  # Infinite for play, 10s for training
  )