import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict

import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict


class Sapien_TEST(gym.Env):

    def __init__(self):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        try:
            self.setup_scene()
            print("\033[32m" + "Render Well" + "\033[0m")
        except Exception as exc:
            raise RuntimeError(f"Render preflight failed: {exc}") from exc

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        self.engine = sapien.Engine()
        render_device_alias = os.environ.get("ROBOTWIN_RENDER_DEVICE")
        if not render_device_alias:
            raise RuntimeError(
                "ROBOTWIN_RENDER_DEVICE must be an explicit PCI alias; "
                "CUDA_VISIBLE_DEVICES does not constrain SAPIEN/Vulkan"
            )
        render_device = sapien.Device(render_device_alias)
        if not render_device.can_render():
            raise RuntimeError(f"Render device cannot render: {render_device_alias}")
        print(
            "ROBOTWIN_RENDER_PREFLIGHT_DEVICE="
            f"{render_device_alias} resolved={render_device} "
            f"pci={render_device.pci_string} cuda_id={render_device.cuda_id}"
        )
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        # Do not instantiate the legacy SapienRenderer here: in SAPIEN 3 it
        # creates a second, default graphics context.  The headless preflight
        # needs only the explicitly pinned RenderSystem below.
        self.renderer = None

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        sapien.physx.set_scene_config(scene_config)
        self.scene = sapien.Scene(
            [
                sapien.physx.PhysxCpuSystem(),
                sapien.render.RenderSystem(render_device),
            ]
        )


if __name__ == "__main__":
    a = Sapien_TEST()
