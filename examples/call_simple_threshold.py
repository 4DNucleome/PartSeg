from PartSegCore.io_utils import load_metadata_base
from PartSegCore.segmentation.segmentation_algorithm import ThresholdPreview
from PartSegImage import GenericImageReader

file_path = "/home/czaki/Dokumenty/smFish/smFISH_7_001_with_points/test.obsep"

profile_str = '{"__SegmentationProfile__": true, "name": "", "algorithm": "Only Threshold", "values": {"channel": 1, "noise_filtering": {"name": "Gauss", "values": {"dimension_type": {"__Enum__": true, "__subtype__": "PartSegCore.segmentation.noise_filtering.DimensionType", "value": 1}, "radius": 1.0}}, "threshold": 300}}'  # noqa: E501
profile = load_metadata_base(profile_str)

image = GenericImageReader.read_image(file_path)

algorithm = ThresholdPreview()
algorithm.set_image(image)
algorithm.set_parameters(profile.values)

res = algorithm.calculation_run(lambda _x, _y: None)
