# API 

## Base classes for creating plugin
The base classes for creating extensions are stored in `partseg_utils.segmentation.algorithm_describe_base.py` 
There are three classes 

*   `AlgorithmProperty` - base class for describing data needed by some algorithm

*   `AlgorithmDescribeBase` - every algorithm need to inherit from this class. It has two methods
    which need to be implemented. Both should be class method  
    
    *   `get_name(cls) -> str` get information about name which should be shown to user
    
    *   `get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]` - get information about 
    parameters which should be provided to algorithm. This data are used to construct input Widget for user. 
    
*   `Register(OrderedDict)` - class used to register ready algorithm. Every group of algorithm should have 
    own instance. To add new element function `register` should be used. This is created to get partial verification
    on start time. 

## Currents algorithms groups

*   Noise removing - `partseg_utils.segmentation.noise_removing`. The register is `noise_removal_dict`
    from this file. This is designed for preprocess step of removing noise from image.
    Currently only "None" and gauss blur are available.
    Other implementations should inherit from `NoiseRemovalBase` from this file.
    They need to implement `noise_remove(cls, chanel: np.ndarray, spacing: typing.Iterable[float], arguments: dict) -> np.ndarray`
    interface where `arguments` contains data defined in `get_fields()`
     
*   Threshold - `partseg_utils.segmentation.threshold`. There are two registers here:

    *   `threshold_dict` - for one threshold algorithm (currently Lower Threshold and Upper threshold).
    Currently it contains manual threshold and automated choose threshold method, like Li, Otsu from SimpleITK library
    
    *   `double_threshold_dict` - for algorithm where part of area need to be decided where it belongs.
    used for `path` and `euclideans` algorithms.
    
*   Segmentation algorithms. - base class for this group is defined in `partseg_utils.segmentation.algorithm_base`
    All method from interface need to be object method.
    To get ability of restarting segmentation without calculating every step the interface contains 5 functions:

*   `set_image(self, image: Image)` - set new image for algorithm

*   `set_mask(self, mask):` - set new mask which limit area of segmentation

*   `set_parameters(self, *args, **kwargs)` - set parameters from `get_fields`

*   `calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult`- 
    the function which is called to run segmentation. It is done is separated thread. 

## Creating Plugins
In current stage PartSeg support plugin only for segmentation methods.
The plugins are loaded before creating objects, so it is possibly to modify its parts with plugins is possible, 
but can brake after change of version.

For binary executable version plugins should be putted in `plugins` folder (like `itk_snap_save`). 
This plugins must export `register()` method. 
To use plugins with PartSeg as python package you need to write custom launcher (base on `Partseg.launcher_main.py`) 
which will register it. There is plan for change it. 

For creating your own plugin most important is module is `PartSeg.utils.register`. Its define `RegisterEnum` and 
function `register(target: Type[AlgorithmDescribeBase], place: RegisterEnum, replace=False)` which should be used 
for registering your plugins in PartSeg. 

All algorithm in PartSeg are class base on `PartSeg.algorthm_describe_base.AlgorithmDescribeBase`, 
but each category has own subclass which extends interface.

Proper class can be obtained form `base_class_dict` from `PartSeg.utils.register` module. Keys in this module are 
values of `RegisterEnum`.
