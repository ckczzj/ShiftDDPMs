import torch.nn as nn

from model.shift_predictor.better_unet import BetterUnet

class LSUNCHURCHShiftPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.better_unet = BetterUnet(
            image_channel=3,
            base_channel=128,
            channel_multiplier=[1, 2, 2, 2],
            num_residual_blocks_of_a_block=2,
            attn_strides=[16],
            attn_heads=1,
            dropout=0.1,
            use_time_embedding=False,
            res_up_down=False,
        )

    def forward(self, img):
        return self.better_unet(img)