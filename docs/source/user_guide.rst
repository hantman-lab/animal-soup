User Guide
==========

The demo notebook is the best place to start: https://github.com/hantman-lab/animal-soup/blob/main/notebooks/demo_notebook.ipynb

This guide provides some more details on the API and concepts for using ``animal-soup``.

Animal-soup interfaces is a collection of "pandas extensions" -- functions that operate on `pandas DataFrames <https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe>`_
and `pandas Series <https://pandas.pydata.org/docs/reference/series.html>`_.

This enables you to create a "psuedo-database" of your behavioral data. No database setup or experience is required, it operates purely on ``pandas`` and standard file systems.

Since this framework uses ``pandas`` extensions, you should be relatively comfortable with basic pandas operations. If you're familiar with ``numpy`` then ``pandas`` will be easy,
here's a quick start guide from the pandas docs: https://pandas.pydata.org/docs/user_guide/10min.html

Accessors and Extensions
========================

There are 4 *accessors* that the ``animal-soup`` API provides, ``behavior``, ``flow_generator``, ``feature_extractor`` and ``sequence``. These allow you to perform operations on a ``pandas.DataFrame``.

Each row in an ``animal-soup`` dataframe corresponds to a **single trial**.

A **single trial** is the combination of:

* animal_id
* session_id
* trial_id
* vid_path
* output_path
* exp_type
* model_params
* notes

**Examples:**

Some common ``behavior`` extensions are:

* ``behavior.add_item()`` - adds a single trial, a single session, or all sessions for an animal to the dataframe
* ``behavior.remove_item()`` - removes a single trial, a single session, or all sessions for an animal from the dataframe
* ``behavior.view()`` - creates a container for viewing your behavioral data and ethograms
* ``behavior.infer()`` - will perform feature extraction and sequence inference to generate predicted dataframes

Some ``flow generator`` extensions:

* ``flow_generator.train()`` - trains a flow generator model based on trials in the dataframe

Some ``feature extractor`` extensions:

* ``feature_extractor.train()`` - trains a feature extractor model based on the trials in the dataframe
* ``feature_extractor.infer()`` - runs feature extractor on a single trial in the dataframe

Some ``sequence`` extensions:

* ``sequence.train()`` - trains a sequence model based on the trials in the current dataframe
* ``sequence.infer()`` - runs sequence model inference to produce a predicted ethograms for a single trial in the dataframe

You must use the appropriate *accessor* on a DataFrame or Series (row) to access the appropriate extension functions. Accessors that operate at the level of the DataFrame can only be referenced using the DataFrame instance.

For example the ``behavior.add_item()`` extension operates on a DataFrame, so you can use it like this:

.. code-block:: python

    # imports
    from animal_soup import *

    # load an existing DataFrame
    df = load_df("/path/to/behavior_dataframe.hdf5")

    # in this case `df` is a DataFrame instance
    # we can use the `behavior` accessor to utilize
    # behavior extensions that operate at
    # the level of the DataFrame

    # for example, ``add_item()`` works at the level of a DataFrame
    df.behavior.add_item(<args>)

In contrast some common extensions, such as ``behavior.infer()`` operate on ``pandas.Series``, i.e. individual DataFrame *rows*. You will need to using indexing on the DataFrame to get the ``pandas.Series`` (row) that you want.

.. code-block:: python

    # imports
    from animal_soup import *

    # load an existing DataFrame
    df = load_df("/path/to/behavior_dataframe.hdf5")

    # df.iloc[n] will return the pandas.Series, i.e. row at the `nth` index

    # we can run feature extractor and sequence inference on the trial in the 0th row
    df.iloc[0].behavior.infer()


**More Examples**

We can run inference on all trials in the dataframe and then create a viewer to look at the predictions.

.. code-block:: python

    # imports
    from animal_soup import *

    # load an existing DataFrame
    df = load_df("/path/to/behavior_dataframe.hdf5")

    for ix, row in df.iterrows():
        row.behavior.infer()

    # create viewer container
    container = df.behavior.view()

    # view the container to see predicted ethograms
    container.view()


Data Management
===============

``animal-soup`` assumes that your behavioral data is stored in the following way:

* animal_id1
    * session_id1
        * trial_id1
        * trial_id2
        ...
    * session_id2
        * trial_id1
        ...
    ...
* animal_id2
    * session_id1
        ...
    ...

Trials in a given session can then be added to the dataframe in a multitude of ways.

1.) Add all sessions for a given animal:

.. code-block:: python

    # imports
    from animal_soup import *

    # create a new dataframe
    df = create_df("/path/to/behavior_dataframe.hdf5")

    # in this case `df` is a DataFrame instance
    # we can use the `behavior` accessor to utilize
    # behavior extensions that operate at
    # the level of the DataFrame

    # for example, ``add_item()`` works at the level of a DataFrame
    df.behavior.add_item(animal_id=<animal_id>)

This will attempt to add all trials in all sessions for the specified animal.

2.) Add a single session for a given animal:

.. code-block:: python

    # assuming use of same dataframe from above
    df.behavior.add_item(animal_id=<animal_id>, session_id=<session_id>)


This will add all trials for the specified session to the dataframe.

3.) Add a single trial to the dataframe.

.. code-block:: python

    # assuming use of same dataframe from above
    df.behavior.add_item(animal_id=<animal_id>, session_id=<session_id>, trial_id=<trial_id>)

This will add a singular trial to the dataframe.

.. note::
    It is not required to specify an experiment type ("table" or "pez") when adding items to the dataframe.
    However, in order to run feature extraction or sequence inference, you must have the experiment type specified so
    that the correct pre-trained model paths can be used. You can always add the experiment type for a given
    trial later, but it is recommended to just pass the experiment type (``exp_type``=<experiment_type>)
    when adding items to the dataframe.

Inference
=========

Once you have added items to a dataframe, you can very easily run inference using a specified ``mode``. The ``mode`` argument
indicates which models to use reconstruct and use for inference. See the table below for information about the models used for each
``mode``.

+--------+-----------------+---------------+-----------------+
| mode   | flow model      | feature model | sequence model  |
+========+=================+===============+=================+
| slow   | TinyMotionNet   | ResNet3D-34   | TGMJ            |
+--------+-----------------+---------------+-----------------+
| medium | MotionNet       | ResNet50      | TGMJ            |
+--------+-----------------+---------------+-----------------+
| fast   | TinyMotionNet3D | ResNet18      | TGMJ            |
+--------+-----------------+---------------+-----------------+

.. note::
    The sequence model used for all ``mode`` types is a TGMJ model. However, it has been specifically trained
    for inference when the features have been extracted using the corresponding flow generator and feature
    extractor.

Here is how you can run inference for a given trial, or your entire dataframe:

.. code-block:: python

    # imports
    from animal_soup import *

    df = load_df('/path/to/dataframe.hdf')

    # top-level folder, all animals/sessions/trials should be directly under this
    set_parent_raw_data_path('/path/to/vids')

    # run inference on entire dataframe
    for ix, row in df.iterrows():
        row.behavior.infer()

    # or run inference on single trial
    df.iloc[0].behavior.infer()

.. note::
    Outputs from running inference get automatically stored to disk in an h5 file. The trial outputs are all stored in a
    single h5 outputs file per session (``<parent_data_path>/<animal_id>/<session_id>/outputs.h5``).

Visualization
=============

Once you have run inference. You can create a viewer to look at the ethograms predictions.

To view predicted ethograms:

.. code-block:: python

    # assuming you have already ran inference like above
    container = df.behavior.view()
    container.show()

.. note::
    The viewer is first returned as a container to provide the user access to elements of the visualization and data
    should they wish to have more control over interacting with their data.

If you wish to edit your predicted ethograms, you can use the interactive ethogram cleaner like so:

.. code-block:: python

    # assuming you have already ran inference like above
    container = df.behavior.clean_ethograms()
    container.show()

This will allow you to edit predicted ethograms in the current dataframe. See the table below for key bindings:

+-----+----------------------------------------------------------------------------------+
| key | action                                                                           |
+=====+==================================================================================+
| 1   | Set indices under selected region for current behavior as occurring, "insert"    |
+-----+----------------------------------------------------------------------------------+
| 2   | Set indices under selected region for current behavior as not occuring, "delete" |
+-----+----------------------------------------------------------------------------------+
| Q   | Change current selected behavior to one above, "move up"                         |
+-----+----------------------------------------------------------------------------------+
| S   | Change current selected behavior to one below, "move down"                       |
+-----+----------------------------------------------------------------------------------+
| R   | Reset ethogram                                                                   |
+-----+----------------------------------------------------------------------------------+
| T   | Reset only current behavior                                                      |
+-----+----------------------------------------------------------------------------------+
| Y   | Save ethogram                                                                    |
+-----+----------------------------------------------------------------------------------+

.. note::
    Any changes to the currently viewed ethogram will be saved automatically. However, you can also
    press the 'Y' key in the event that you manually change values in the ethogram
    and want them to be saved.




