from project_utils.segmentation.restartable_segmentation_algorithms import final_algorithm_list

part_algorithm_dict = dict(((x.get_name(), x) for x in final_algorithm_list))


class SegmentationProfile(object):
    def __init__(self, name, algorithm, values):
        self.name = name
        self.algorithm = algorithm
        self.values = values

    def __str__(self):
        return "Segmentation profile name: " + self.name + "\nAlgorithm: " + self.algorithm + "\n" + "\n".join(
            [f"{k.replace('_', ' ')}: {v}" for k, v in self.values.items()])

    def __repr__(self):
        return f"SegmentationProfile(name={self.name}, algorithm={self.algorithm}, values={self.values})"
