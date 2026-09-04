# Vendored RoboTwin

This directory vendors code from the upstream RoboTwin repository:

- Upstream project: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT License

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboTwin remain subject to the MIT License in this directory.
- For this project, the only locally maintained policy implementation is
  `experiments/robotwin/fastwam_policy` in the FastWAM repository.
- `policy/fastwam_policy` is a runtime-only symlink to that implementation; it
  is not a second maintained copy.
- If code is later copied from any upstream subdirectory with an additional license notice, the corresponding license file and attribution must also be preserved.

Local modifications:

- RoboTwin is vendored under `third_party/RoboTwin` for easier integration with this project.
- Unused upstream policy implementations under `policy/` may be removed for redistribution.
- Additional local changes may be applied to adapt RoboTwin to this repository.
- This directory is the only maintained RoboTwin **code and task-config source of truth**
  for FastWAM inference.  The separate checkout at
  `/local_home/zhaizicheng/fastwam_robotwin/RoboTwin` is not imported or patched.
- On this machine, `assets` is a local symlink to that checkout's downloaded
  asset bundle.  It is an asset donor only; task implementations, task configs,
  and evaluation code must not be maintained there.
