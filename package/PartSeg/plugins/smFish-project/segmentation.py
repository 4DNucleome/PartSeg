import operator
import typing
from copy import deepcopy
from typing import Callable

import numpy as np
import SimpleITK

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from PartSegCore.channel_class import Channel
from PartSegCore.segmentation import SegmentationAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, SegmentationResult
from PartSegCore.segmentation.noise_filtering import NoneNoiseFiltering, noise_filtering_dict
from PartSegCore.segmentation.threshold import BaseThreshold, threshold_dict


class SMSegmentation(SegmentationAlgorithm):
    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        channel_nuc = self.get_channel(self.new_parameters["channel_nuc"])
        noise_filtering_parameters = self.new_parameters["noise_filtering_nucleus"]
        cleaned_image = noise_filtering_dict[noise_filtering_parameters["name"]].noise_filter(
            channel_nuc, self.image.spacing, noise_filtering_parameters["values"]
        )
        thr: BaseThreshold = threshold_dict[self.new_parameters["nucleus_threshold"]["name"]]
        nucleus_mask, nucleus_thr_val = thr.calculate_mask(
            cleaned_image, self.mask, self.new_parameters["nucleus_threshold"]["values"], operator.ge
        )
        nucleus_connect = SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(nucleus_mask), True)
        nucleus_segmentation = SimpleITK.GetArrayFromImage(
            SimpleITK.RelabelComponent(nucleus_connect, self.new_parameters["minimum_nucleus_size"])
        )

        channel_molecule = self.get_channel(self.new_parameters["channel_molecule"]).astype(np.float32)
        mask = self.mask if self.mask is not None else channel_molecule > 0
        mean_background = np.mean(channel_molecule[mask > 0])
        channel_molecule[mask == 0] = mean_background
        if NoneNoiseFiltering.get_name() != self.new_parameters["background_estimate"]:
            background_estimate_parameters = self.new_parameters["background_estimate"]
            background = noise_filtering_dict[background_estimate_parameters["name"]].noise_filter(
                channel_molecule, self.image.spacing, background_estimate_parameters["values"]
            )
        else:
            background = np.full(
                channel_molecule.shape,
                mean_background,
                dtype=(mean_background),
            )

        foreground_estimate_parameters = self.new_parameters["foreground_estimate"]
        foreground = noise_filtering_dict[foreground_estimate_parameters["name"]].noise_filter(
            channel_molecule, self.image.spacing, foreground_estimate_parameters["values"]
        )
        estimated = foreground - background
        thr: BaseThreshold = threshold_dict[self.new_parameters["molecule_threshold"]["name"]]
        molecule_mask, molecule_thr_val = thr.calculate_mask(
            estimated, self.mask, self.new_parameters["molecule_threshold"]["values"], operator.ge
        )
        nucleus_connect = SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(molecule_mask), True)

        molecule_segmentation = SimpleITK.GetArrayFromImage(
            SimpleITK.RelabelComponent(nucleus_connect, self.new_parameters["minimum_molecule_size"])
        )

        sizes = np.bincount(molecule_segmentation.flat)
        elements = np.unique(molecule_segmentation[molecule_segmentation > 0])

        cellular_components = set(np.unique(molecule_segmentation[nucleus_segmentation == 0]))
        if 0 in cellular_components:
            cellular_components.remove(0)
        nucleus_components = set(np.unique(molecule_segmentation[nucleus_segmentation == 1]))
        if 0 in nucleus_components:
            nucleus_components.remove(0)
        mixed_components = cellular_components & nucleus_components
        cellular_components = cellular_components - mixed_components
        nucleus_components = nucleus_components - mixed_components
        label_types = {}
        label_types.update({i: "Nucleus" for i in nucleus_components})
        label_types.update({i: "Cytoplasm" for i in cellular_components})
        label_types.update({i: "Mixed" for i in mixed_components})

        annotation = {el: {"voxels": sizes[el], "type": label_types[el], "number": el} for el in elements}
        position_masking = np.zeros(elements.max() + 1, dtype=molecule_segmentation.dtype)
        for el in cellular_components:
            position_masking[el] = 1
        for el in mixed_components:
            position_masking[el] = 2
        for el in nucleus_components:
            position_masking[el] = 3
        position_array = position_masking[molecule_segmentation]

        return SegmentationResult(
            roi=molecule_segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "nucleus segmentation": AdditionalLayerDescription(data=nucleus_segmentation, layer_type="labels"),
                "roi segmentation": AdditionalLayerDescription(data=molecule_segmentation, layer_type="labels"),
                "estimated signal": AdditionalLayerDescription(data=estimated, layer_type="image"),
                "background": AdditionalLayerDescription(data=background, layer_type="image"),
                "channel molecule": AdditionalLayerDescription(data=channel_molecule, layer_type="image"),
                "position": AdditionalLayerDescription(data=position_array, layer_type="labels"),
            },
            roi_annotation=annotation,
            alternative_representation={"position": position_array},
        )

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), deepcopy(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "sm-fish segmentation"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("channel_nuc", "Nucleus Channel", 0, property_type=Channel),
            AlgorithmProperty(
                "noise_filtering_nucleus",
                "Filter nucleus",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "nucleus_threshold",
                "Nucleus Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_nucleus_size", "Minimum nucleus size (px)", 500, (0, 10 ** 6), 1000),
            AlgorithmProperty("channel_molecule", "Channel molecule", 1, property_type=Channel),
            AlgorithmProperty(
                "background_estimate",
                "Background estimate",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "foreground_estimate",
                "Foreground estimate",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "molecule_threshold",
                "Molecule Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_molecule_size", "Minimum molecule size (px)", 5, (0, 10 ** 6), 1000),
        ]
