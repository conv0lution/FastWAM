from pathlib import Path

from experiments.asre_diagnosis.g0.run_g0 import _base_environment


def test_base_environment_keeps_cuda_and_egl_on_the_rendering_gpu(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")

    environment = _base_environment(tmp_path, rendering_gpu=4)

    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "4"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "4"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
