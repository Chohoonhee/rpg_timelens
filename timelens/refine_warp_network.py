import torch as th
from timelens.common import warp
# from timelens import fusion_network, warp_network
from timelens.superslomo import unet
from timelens.common import pytorch_tools
from torch import nn


def _pack_for_residual_flow_computation(example):
    tensors = [
        example["middle"]["{}_warped".format(packet)] for packet in ["after", "before"]
    ]
    tensors.append(example["middle"]["fusion"])
    return th.cat(tensors, dim=1)


def _pack_images_for_second_warping(example):
    return th.cat(
        [example["middle"]["after_warped"], example["middle"]["before_warped"]],
    )


def _pack_output_to_example(example, output):
    (
        example["middle"]["before_refined_warped"],
        example["middle"]["after_refined_warped"],
        example["middle"]["before_refined_warped_invalid"],
        example["middle"]["after_refined_warped_invalid"],
        example["before"]["residual_flow"],
        example["after"]["residual_flow"],
    ) = output

def _pack(example):
    return th.cat([example['before']['voxel_grid'],
                   example['before']['rgb_image_tensor'],
                   example['after']['voxel_grid'],
                   example['after']['rgb_image_tensor']], dim=1)


def _pack_voxel_grid_for_flow_estimation(example):
    return th.cat(
        [example["before"]["reversed_voxel_grid"], example["after"]["voxel_grid"]]
    )


def _pack_images_for_warping(example):
    return th.cat(
        [example["before"]["rgb_image_tensor"], example["after"]["rgb_image_tensor"]]
    )


def _pack_output_to_example(example, output):
    (
        example["middle"]["before_warped"],
        example["middle"]["after_warped"],
        example["before"]["flow"],
        example["after"]["flow"],
        example["middle"]["before_warped_invalid"],
        example["middle"]["after_warped_invalid"],
    ) = output


class Warp(nn.Module):
    def __init__(self):
        super(Warp, self).__init__()
        self.flow_network = unet.UNet(5, 2, False)

    def from_legacy_checkpoint(self, checkpoint_filename):
        checkpoint = th.load(checkpoint_filename)
        self.load_state_dict(checkpoint["networks"])

    def run_warp(self, example):
        flow = self.flow_network(_pack_voxel_grid_for_flow_estimation(example))
        warped, warped_invalid = warp.backwarp_2d(
            source=_pack_images_for_warping(example),
            y_displacement=flow[:, 0, ...],
            x_displacement=flow[:, 1, ...],
        )
        (before_flow, after_flow) = th.chunk(flow, chunks=2)
        (before_warped, after_warped) = th.chunk(warped, chunks=2)
        (before_warped_invalid, after_warped_invalid) = th.chunk(
            warped_invalid.detach(), chunks=2
        )
        return (
            before_warped,
            after_warped,
            before_flow,
            after_flow,
            before_warped_invalid,
            after_warped_invalid,
        )
    
    def run_and_pack_to_example(self, example):
        _pack_output_to_example(example, self.run_warp(example))

    def forward(self, example):
        return self.run_warp(example)

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        
    def run_fusion(self, example):
        return self.fusion_network(_pack(example))
        
    def from_legacy_checkpoint(self, checkpoint_filename):
        checkpoint = th.load(checkpoint_filename)
        self.load_state_dict(checkpoint["networks"])

    def run_and_pack_to_example(self, example):
        example['middle']['fusion'] = self.run_fusion(example)
        
    def forward(self, example):
        return self.run_fusion(example)


class RefineWarp(Warp, Fusion):
    def __init__(self):
        Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)

    def run_refine_warp(self, example):
        Warp.run_and_pack_to_example(self, example)
        Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = th.chunk(residual, 2, dim=1)
        residual = th.cat([after_residual, before_residual], dim=0)
        refined, refined_invalid = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )
        
        (after_refined, before_refined) = th.chunk(refined, 2)
        (after_refined_invalid, before_refined_invalid) = th.chunk(
            refined_invalid.detach(), 2)
        return (
            before_refined,
            after_refined,
            before_refined_invalid,
            after_refined_invalid,
            before_residual,
            after_residual,
        )


    def run_fast(self, example):
        Warp.run_and_pack_to_example(self, example)
        Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = th.chunk(residual, 2, dim=1)
        residual = th.cat([after_residual, before_residual], dim=0)
        refined, _ = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )

        return th.chunk(refined, 2)

    def run_and_pack_to_example(self, example):
        _pack_output_to_example(example, self.run_refine_warp(example))

    def forward(self, example):
        return self.run_refine_warp(example)
