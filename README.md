# icub_mjlab: iCub Robot Learning in MuJoCo with mjlab

Entorno de Aprendizaje por Refuerzo (RL) para el robot humanoide **iCub** basado en **[mjlab](https://github.com/mujocolab/mjlab)** y acelerado por **MuJoCo Warp** en GPU.

Este repositorio implementa la tarea de manipulación y elevación de cubos sobre una mesa de trabajo (`Mjlab-Lift-Cube-iCub`), siguiendo la arquitectura *manager-based* de Isaac Lab adaptada a MuJoCo.

---

## 🚀 Requisitos Previos

- Sistema operativo Linux (probado en Ubuntu).
- GPU NVIDIA compatible con CUDA.
- [uv](https://docs.astral.sh/uv/) instalado.
- Repositorio principal de `mjlab` instalado o disponible en tu entorno.

---

## 📦 Instalación

1. **Clonar este repositorio con sus submódulos:**
   ```bash
   git clone --recurse-submodules https://github.com/mrojasanchez/icub_mjlab.git
   cd icub_mjlab
   ```

   > **Nota:** Si ya lo clonaste sin submódulos, inicialízalos con:
   > ```bash
   > git submodule update --init --recursive
   > ```

2. **Instalar el entorno y dependencias con `uv`:**
   ```bash
   uv sync
   uv pip install -e .
   ```

---

## 🎮 Uso y Simulación

### 1. Visualizar el entorno (Modo Play)

- **Robot en reposo (Pose Home):**
  ```bash
  uv run play Mjlab-Lift-Cube-iCub --agent zero
  ```

- **Robot con acciones aleatorias:**
  ```bash
  uv run play Mjlab-Lift-Cube-iCub --agent random
  ```

- **Visualizar un solo entorno (sin flechas de selección):**
  ```bash
  uv run play Mjlab-Lift-Cube-iCub --agent zero --num-envs 1
  ```

### 2. Entrenar la Política con RL (PPO)

```bash
uv run train Mjlab-Lift-Cube-iCub
```

---

## 📂 Estructura del Proyecto

```text
icub_mjlab/
├── pyproject.toml         # Configuración del paquete y entry point para mjlab.tasks
├── README.md              # Documentación del proyecto
├── src/
│   └── icub_mjlab/
│       ├── __init__.py    # Registro de la tarea Mjlab-Lift-Cube-iCub
│       ├── env_cfg.py     # Definición del MDP (escena, recompensas, observaciones, eventos)
│       ├── icub_constants.py # Cinemática, límites articulares y calibración de acciones
│       ├── rl_cfg.py      # Hiperparámetros de entrenamiento PPO (rsl_rl)
│       ├── table.xml      # Modelo XML de la mesa de trabajo con materiales
│       └── assets/        # Submódulo con modelos MJCF, mallas STL y texturas del iCub
```

---

## 📜 Licencia y Atribuciones

- Modelos y geometrías de iCub provistos por [CAMADA-UTFSM/icub_mj_model](https://github.com/CAMADA-UTFSM/icub_mj_model).
- Integración de aprendizaje robótico desarrollada con el framework [mjlab](https://github.com/mujocolab/mjlab).
