from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam import FastWAM


class DevicePlacementTest(unittest.TestCase):
    def test_explicit_text_encoder_device_survives_model_to(self) -> None:
        model = FastWAM.__new__(FastWAM)
        nn.Module.__init__(model)
        model.mot = nn.Linear(2, 2, bias=False)
        model.vae = nn.Linear(2, 2, bias=False)
        model.text_encoder = nn.Linear(2, 2, bias=False)
        model.text_encoder_device = torch.device("cpu")

        model.to(dtype=torch.float64)

        self.assertEqual(model.mot.weight.dtype, torch.float64)
        self.assertEqual(model.vae.weight.dtype, torch.float64)
        # An explicitly sharded encoder follows its own placement request and
        # is not traversed by the main model's recursive .to(...).
        self.assertEqual(model.text_encoder.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
