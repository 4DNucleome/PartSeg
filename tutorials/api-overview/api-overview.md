# API 

## Base classes for creating plugin
The base classes for creating extensions are stored in `project_utils.segmentation.algorithm_describe_base.py` 
There are three classes 

* `AlgorithmProperty` - base class for describing data needed by some algorithm
* `AlgorithmDescribeBase` - every algorithm need to inherit from this class. It has two methods
which need to be implemented. Both should be class method  
    * `get_name(cls) -> str` get information about name which should be shown to user
    * `get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]` - get information about 
    parameters which should be provided to algorithm. This data are used to construct input Widget for user. 
* `Register(OrderedDict)` - class used to register ready algorithm. Every group of algorithm should have 
own instance. To add new element function `register` should be used. This is created to get partial verification 
on start time. 

## Currents algorithms groups

* Noise removing - `project_utils.segmentation.noise_removing`. The register is `noise_removal_dict`
from this file. This is designed for preprocess step of removing noise from image.
Currently only "None" and gauss blur are available. 
Other implementations should inherit from `NoiseRemovalBase` from this file. 
They need to implement `noise_remove(cls, chanel: np.ndarray, spacing: typing.Iterable[float], arguments: dict) -> np.ndarray`
interface where `arguments` contains data defined in `get_fields()` 
* Threshold - `project_utils.segmentation.threshold`. There are two registers here:
    * `threshold_dict` - for one  

