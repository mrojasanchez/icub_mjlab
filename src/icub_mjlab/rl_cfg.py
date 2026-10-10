"""RL configuration for DextrAH-RGB Teacher FGP in icub_mjlab."""

from mjlab.rl.config import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def liftcube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for DextrAH-RGB Teacher FGP.

  Asymmetric Actor-Critic recurrent architecture following DextrAH-RGB:
  - Actor: LSTM 512 + MLP [512, 512] with ELU activation
  - Critic: LSTM 2048 + MLP [1024, 512] with ELU activation
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="RNNModel",
      rnn_type="lstm",
      rnn_hidden_dim=512,
      rnn_num_layers=1,
      hidden_dims=(512, 512),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_range": (0.05, 1.0),
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="RNNModel",
      rnn_type="lstm",
      rnn_hidden_dim=2048,
      rnn_num_layers=1,
      hidden_dims=(1024, 512),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.015,
      max_grad_norm=1.0,
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    logger="tensorboard",
    experiment_name="dextrah_teacher_fgp",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=4_000,
  )