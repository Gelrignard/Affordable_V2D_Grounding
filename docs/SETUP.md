# CPU setup

Run commands from the repository root. Policy training has its own [GPU runtime setup](training/POLICY_HANDOFF.md).

## Dynamics correction

With Python 3.11 or later:

```text
python -m pip install -r correction/requirements.txt
python -m correction.scripts.demo
```

The standalone contact optimizer uses numerical dynamics and optimization libraries. Robot retargeting additionally needs Pinocchio, Pink, and the robot assets below.

## Robot retargeting

Create a CPU environment with Micromamba or Conda. The supplied lock file records Windows package builds; use the environment specification to resolve packages on other platforms.

```text
micromamba create -y -p ./.retarget-env -f setup/preview-environment.yml
micromamba run -p ./.retarget-env python -m pip install -r requirements-cpu.txt
micromamba run -p ./.retarget-env python -m pip install --no-deps pin-pink==4.2.0
micromamba run -p ./.retarget-env python -m pip install --no-deps -e ./retargeting
```

Prepare the [pinned robot dependency and assets](NVIDIA_COMPATIBILITY.md), then follow the [correction pipeline](correction/PIPELINE.md). Decode SOMA inputs in the separate [SOMA environment](SOMA_SETUP.md).

## Generated artifacts

Use an output directory outside the source packages or under ignored `outputs/`. Dataset caches, Python environments, weights, checkpoints, and run reports remain local. The public repository contains reusable implementation and configuration files.
