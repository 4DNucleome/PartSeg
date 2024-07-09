Creating plugins for PartSeg
============================

PartSeg has a plugin system, but because of a lack of documentation, it is not very popular.
This document is an attempt to explain how to create plugins for PartSeg. The list of good plugins that could be inspirration
is available at the :ref:`end of this document<Already existing plugins>`.

.. note::

    This document needs to be completed and may need to be corrected. If you have any questions, please inform me by a GitHub issue.

PartSeg plugin system was designed at the beginning of the project when PartSeg was a python 2.7 project, and still, not
all possibilities given by recent python versions are used. For example, plugin methods are still class-based,
but there is a plan to allow function-based plugins.

Where plugin could contribute in PartSeg
----------------------------------------

Here I describe nine areas where plugins could contribute:


* **Threshold algorithms** - algorithms for binarizing single channel data.

* **Noise Filtering algorithms** - algorithms for filtering noise from a single channel of data.

* **Flow algorithms** - Watershed-like algorithms for flow from core object to the whole region.
  Currently, they are used in "Lower Threshold with Watershed" and "Upper Threshold with Watershed"
  segmentation methods

* **ROI Analysis algorithm** - algorithm available in *ROI Analysis* GUI for ROI extraction.

* **ROI Analysis load method** - method to load data in *ROI Analysis* GUI.

* **ROI Analysis save method** - method to save data in *ROI Analysis* GUI.

* **ROI Mask algorithm** - algorithm available in *ROI Mask* GUI for ROI extraction.

* **ROI Mask load method** - method to load data in *ROI Mask* GUI.

* **ROI Mask save method** - method to save data in *ROI Mask* GUI.


A person interested in developing a plugin could wound the whole list in :py:mod:`PartSegCore.register` module.

Plugin detection
----------------

PartSeg uses :py:func:`pkg_resources.iter_entry_points` module to detect plugins.
To be detected, the plugin needs to provide one of ``partseg.plugins``, ``PartSeg.plugins``.
``partsegcore.plugins`` or ``PartSegCore.plugins`` entry point in python package metadata.
The example of ``setup.cfg`` configuration from a plugin could be found  `here <https://github.com/Czaki/Trapalyzer/blob/fc5b84fde2fb1fe4bea75bdd1e4a483772115500/setup.cfg#L43>`_




``--develop`` mode of PartSeg
-----------------------------

PartSeg allows to use of plugins in ``--develop`` mode. In this mode is a settings dialog the additional
tab "Develop" is added. In this tab, there is Reload button. After the button press, PartSeg tries
to reload all plugins. This feature allows the development of a plugin without the need of too often restarting of PartSeg.

This approach is limited and reloads only entries pointed in entry points.
It is the plugin creator's responsibility to reload all other modules required by the plugin.
To detect if the file is during reload, the plugin author could use the following code:


.. code-block:: python

    try:

        reloading
    except NameError:
        reloading = False
    else:
        reloading = True

The ``importlib.reload`` function could be used to reload required modules.

.. code-block:: python

    import importlib
    importlib.reload(module)


Already existing plugins
------------------------
Here we list already existing PartSeg plugins. All plugins are also available when using PartSeg as a napari plugin.
New plugin creator may want to look at them to see how to create new plugins.

PartSeg-smfish
~~~~~~~~~~~~~~
This is plugin for processing smFISH data. It could be found under https://github.com/4DNucleome/PartSeg-smfish/ page.
This plugin provides a custom segmentation algorithm for smfish data (inspired by bigFISH algorithm that does not work well for our data)
and custom measurement methods.

The plugin is available from pypi and conda.

PartSeg-bioimageio
~~~~~~~~~~~~~~~~~~
PartSeg plugin to run bioimage.io deep learning models. It could be found under https:/github.com/czaki/PartSeg-bioimageio/ page
This plugin allows to selection model saved in bioimage.io format from the disc and runs it on selected data with the test this model in interactive mode.

As it depends on deep learn libraries, it cannot be used in PartSeg binary distribution.

In this plugin, plugin creator could see an example of using custom magicgui-based widget for selecting a model.

The plugin is under active development and currently available only on GitHub.

Trapalyzer
~~~~~~~~~~

This is plugin developed for process neutrophile `data <https://zenodo.org/record/7335168>`_.
It provides custom segmentation data to find multiple class of cells in one run and custom measurement methods.

It could be found on github https://github.com/Czaki/Trapalyzer and pypi.
