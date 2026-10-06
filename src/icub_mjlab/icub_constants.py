from pathlib import Path
import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
##
# MJCF and assets.
##

_HERE = Path(__file__).parent

_ICUB_XML: Path = _HERE / "assets" / "xml" / "icub_visuomanip_ikin_limits.xml"
assert _ICUB_XML.exists()

def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_ICUB_XML))

_JOINT_ACTUATOR_TARGETS = (
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "l_wrist_prosup",
    "l_wrist_pitch",
    "l_wrist_yaw",
    "l_hand_thumb_0_joint",
    "l_hand_thumb_1_joint",
    "l_hand_index_1_joint",
    "l_hand_middle_1_joint",
    "neck_pitch",
    "neck_roll",
    "neck_yaw",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup",
    "r_wrist_pitch",
    "r_wrist_yaw",
    "r_hand_thumb_0_joint",
    "r_hand_thumb_1_joint",
    "r_hand_index_1_joint",
    "r_hand_middle_1_joint",
)

_TENDON_ACTUATOR_TARGETS = (
    "l_hand_finger_tendon",
    "l_thumb_distal_tendon",
    "l_index_distal_tendon",
    "l_middle_distal_tendon",
    "l_pinky_tendon",
    "r_hand_finger_tendon",
    "r_thumb_distal_tendon",
    "r_index_distal_tendon",
    "r_middle_distal_tendon",
    "r_pinky_tendon",
)

# Tell mjlab to use the motors defined in the XML as is.
_ICUB_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=_JOINT_ACTUATOR_TARGETS,
            transmission_type=TransmissionType.JOINT,
        ),
        XmlActuatorCfg(
            target_names_expr=_TENDON_ACTUATOR_TARGETS,
            transmission_type=TransmissionType.TENDON,
        ),
    ),
)

def _extract_home_joint_pos() -> dict[str, float]:
    spec = mujoco.MjSpec.from_file(str(_ICUB_XML))
    model = spec.compile()
    return {
        j.name: float(model.key("home").qpos[model.joint(j.name).qposadr[0]])
        for j in spec.joints
    }

_HOME_JOINT_POS = _extract_home_joint_pos()

_HOME_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.65),
    joint_pos=_HOME_JOINT_POS,
    joint_vel={".*": 0.0},
)

# Bundle the spec loader, actuator, and initial state into one config.
def get_icub_robot_cfg(swing_up: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=_HOME_INIT,
        spec_fn=_get_spec,
        articulation=_ICUB_ARTICULATION,
    )

HOME_INIT = _HOME_INIT
_get_icub_robot_cfg = get_icub_robot_cfg


def _extract_action_scale_and_clip(
    scale_factor: float = 0.1,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
  spec = mujoco.MjSpec.from_file(str(_ICUB_XML))
  model = spec.compile()
  scale_dict = {}
  clip_dict = {}
  for name in _JOINT_ACTUATOR_TARGETS:
    jnt = model.joint(name)
    r_min, r_max = float(jnt.range[0]), float(jnt.range[1])
    clip_dict[name] = (r_min, r_max)
    scale_dict[name] = round(scale_factor * (r_max - r_min), 4)
  return scale_dict, clip_dict


ICUB_ACTION_SCALE, ICUB_ACTION_CLIP = _extract_action_scale_and_clip()


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_icub_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)

  viewer.launch(model, data)