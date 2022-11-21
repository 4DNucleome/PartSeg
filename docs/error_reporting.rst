Error reporting
===============

.. important::
  In the ideal state, during using PartSeg, there should be no exceptions.
  Only dialogs with information about what the user has done wrong and how to fix it.
  So If you meet the exception, please report it to the developers.

We created the PartSeg software to simplify sharing segmentation algorithms,
also with non-programming persons. Because of this, next to the standard way of
error reporting using `GitHub issues <https://github.com/4DNucleome/PartSeg/issues>`_
it is possible to report the error using the Report error button in the error dialog.

.. image:: images/error_dialog.png
   :width: 600
   :alt: Error dialog

It is possible to disable error reporting using sentry in two ways.
The first is to launch PartSeg using the ``--no_report flag``.
The second is to set the ``PARTSEG_SENTRY_URL`` environment variable to
the empty string. This second way allows a user to overwrite the sentry
URL and collect error reports in a custom sentry instance.




Information stored in an error report
-------------------------------------
Error report contains information about:

* PartSeg version
* Python version
* Operating system
* List of installed packages
* Stacktrace with variables values (with the full path to files)
* Computer name
* User name

We decided to collect this information based on collaboration with non-coding users.
Information about the computer name and user name are collected to find proper error
reports when got an email from such a user
or reach the user to collect additional information required to solve the problem.
Without initiative from the user, collected data does not allow to make contact from the developer side.

If you prefer to keep this information private, you could still
open github issue and provide information about your problem.


Control of error reporting from code
------------------------------------

It is possible to control error reporting using variables from :py:mod:`PartSeg.state_store`.
PartSeg uses :py:func:`PartSeg.common_backend.except_hook.my_excepthook` as the exception hook.


Data retention
--------------

Sentry is removing reports after 90 days.
Here are details in their `documentation <https://sentry.io/security/#data-retention>`_.
