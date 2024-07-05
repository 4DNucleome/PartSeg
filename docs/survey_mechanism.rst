PartSeg surveys
===============

Since the partseg 0.15.3 release there is a new feature that allows
to ask users for fill in the PartSeg application.

How it works
------------

Whole logic is implemented in ``PartSeg._launcher.check_survey`` module.

.. image:: images/survey.png
   :width: 600
   :alt: Survey message

1. Check if ignore file exist and is last modified less than 16 hours ago. If yes, do nothing.
2. Fetch data from https://raw.githubusercontent.com/4DNucleome/PartSeg/develop/survey_url.txt.
   If file do not exists or is empty, do nothing.
   Save fetched data as url to form.
3. Check if ignore file exist and its content is equal to fetched url. If yes, do nothing.
4. Display message to user with information that there is a survey available.

   * If user click "Open survey" button, open browser with fetched url.
   * If user click "Ignore" button, save fetched url to ignore file.
   * If user click "Close" button, touch the ignore file to prevent showing the message again for 16 hours.

.. graphvix::

    digraph G {
        "Check if ignore file exist and is last modified less than 16 hours ago" -> "Fetch data from https://raw.githubusercontent.com/4DNucleome/PartSeg/develop/survey_url.txt"
        "Fetch data from https://raw.githubusercontent.com/4DNucleome/PartSeg/develop/survey_url.txt" -> "Check if ignore file exist and its content is equal to fetched url"
        "Check if ignore file exist and its content is equal to fetched url" -> "Display message to user with information that there is a survey available"
        "Display message to user with information that there is a survey available" -> "If user click 'Open survey' button, open browser with fetched url"
        "Display message to user with information that there is a survey available" -> "If user click 'Ignore' button, save fetched url to ignore file"
        "Display message to user with information that there is a survey available" -> "If user click 'Close' button, touch the ignore file to prevent showing the message again for 16 hours"
    }
