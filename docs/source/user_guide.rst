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


In order for ``animal_soup`` to find your data you must set the parent data path using ``set_parent_raw_data_path()``.

This function (modeled from ``mesmerize_core``) sets the top level raw data directory. This should be set to the top level directory where your behavioral data is stored.
This allows you to move your behavioral data directory structure between computers, as long as you keep everything under the parent path the same.

Trials in a given session can then be added to the dataframe in a multitude of ways.

.. note::
    For each trial, there should be a front and side video that will get concatenated together for you on the fly during inference
    and for visualizations.

1.) Add all sessions for a given animal:

.. code-block:: python

    # imports
    from animal_soup import *

    set_parent_raw_data_path('/path/to/folder/above/behavior/data')

    # create a new dataframe
    df = create_df("/path/to/behavior_dataframe.hdf5")

    # in this case `df` is a DataFrame instance
    # we can use the `behavior` accessor to utilize
    # behavior extensions that operate at
    # the level of the DataFrame

    # for example, ``add_item()`` works at the level of a DataFrame
    df.behavior.add_item(animal_id='my_animal_id')

This will attempt to add all trials in all sessions for the specified animal.

2.) Add a single session for a given animal:

.. code-block:: python

    # assuming use of same dataframe from above
    df.behavior.add_item(animal_id='my_animal_id', session_id='my_session_id')


This will add all trials for the specified session to the dataframe.

3.) Add a single trial to the dataframe.

.. code-block:: python

    # assuming use of same dataframe from above
    df.behavior.add_item(animal_id='my_animal_id', session_id='my_session_id', trial_id='my_trial_id')

This will add a singular trial to the dataframe.

.. note::
    It is not required to specify an experiment type ("table" or "pez") when adding items to the dataframe.
    However, in order to run feature extraction or sequence inference, you must have the experiment type specified so
    that the correct pre-trained model paths can be used. You can always add the experiment type for a given
    trial later, but it is recommended to just pass the experiment type (``exp_type='table'``)
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

Customization/Extension
=======================

``animal-soup`` has been designed under the assumption that you will not need to re-train any of the default
models that come with the package for the Hantman Lab reach-to-grab task (regardless of experiment type: table, pez, taz, etc.).

However, in the event that you would like to further customize the models that you are using for inference,
the information below will explain how to do so:

.. note::
    If you are unfamiliar with the model structure of ``animal-soup`` and the way in which behavioral inference is done,
    please see the **Background** page of the docs before continuing!

Using Your Own Model Checkpoints - Training
-------------------------------------------

**Flow Generator**

When training the flow generator, you must specify a ``mode`` (“slow”, “medium”, or “fast”).
The ``mode`` argument indicates which type of flow generator model to construct (TinyMotionNet3D, MotionNet, or TinyMotionNet).

+--------+-----------------+
| mode  | model            |
+========+=================+
| fast   | TinyMotionNet   |
+--------+-----------------+
| medium | MotionNet       |
+--------+-----------------+
| slow   | TinyMotionNet3D |
+--------+-----------------+

For each ``mode``, there is a pre-trained model checkpoint that can be loaded. However, if you
have already trained the flow generator previously, you can use the ``model_in`` kwarg to specify a path
to a flow generator model checkpoint. This will allow you to start flow generator training from that checkpoint as opposed
to a pre-trained model checkpoint.

**If you are using a checkpoint specified by** ``model_in``
**, the** ``mode`` **argument must match the type of model that the checkpoint is for.**

For example, if you previously trained the flow generator with ``mode=’slow’``, then the checkpoint saved from training is for a TinyMotionNet3D model. Therefore, if you go to use that checkpoint for training in the future,
then you will need to make sure the ``mode`` argument is “slow” otherwise you will get errors when trying to reconstruct the appropriate flow generator model training.

.. code-block:: python

    # model output path where you want to store training results
    output_path = "/path/to/model/outputs"
    # dateframe you want to use to train the flow generator
    df.flow_generator.train(mode="slow", model_out=output_path)

    # now say you have a second dateframe and you want to train the
    # flow generator using the checkpoint generated from the previous training above
    df2.flow_generator.train(mode="slow", model_in=output_path)


**Feature Extractor**

When training the feature extractor, you must also specify a ``mode`` ("slow", "medium", or "fast").
The ``mode`` argument indicates which type of feature extractor generator model to construct (ResNet3D_34, ResNet50, or ResNet18)
as well as which flow generator model to construct (TinyMotionNet3D, MotionNet, TinyMotionNet.

For each ``mode``, there is a pre-trained model checkpoint that can be loaded for the feature extractor and flow generator. However, if you
have already trained the flow generator or feature extractor previously, you can specify paths to those checkpoints.
This will allow you to start feature extractor training from that checkpoint as opposed to a pre-trained model checkpoint.

**If you are specifying checkpoint paths for the flow generator and feature extractor they must be to model checkpoints that match the same mode.**

+--------+-----------------+---------------+
| mode   | flow model      | feature model |
+========+=================+===============+
| slow   | TinyMotionNet   | ResNet3D-34   |
+--------+-----------------+---------------+
| medium | MotionNet       | ResNet50      |
+--------+-----------------+---------------+
| fast   | TinyMotionNet3D | ResNet18      |
+--------+-----------------+---------------+

Due to the architectures of the models, you must retain the same ``mode`` through training/inference.

To specify a flow generator model checkpoint you can specify a checkpoint path using the ``flow_model_in`` kwarg.
You can specify a feature model checkpoint for reconstructing the feature extractor using the ``feature_model_in`` kwarg.

If the ``mode`` arg provided does not match the model types that the checkpoints are for as stated in the above table, you will
get errors trying to create the flow generator and feature extractor.

.. code-block:: python

    # paths to previous model checkpoints
    # for example, assume these were previously trained with mode='slow'
    flow_checkpoint = '/path/to/flow/generator/checkpoint.cpkt'
    feature_checkpoint = '/path/to/feature/extractor/checkpoint.cpkt'

    # dataframe for training the feature extractor
    df.feature_extractor.train(mode="slow", flow_model_in=flow_checkpoint, feature_model_in=feature_checkpoint)

    # could also train the feature extractor without having flow generator checkpoint
    # will simply use default pre-trained flow generator checkpoint
    df.feature_extractor.train(mode="slow", feature_model_in=feature_checkpoint)

**Sequence Model**

When training the sequence model, you must also specify a ``mode`` ("slow", "medium", or "fast").
The ``mode`` argument indicates which type of sequence model to construct based on the ``mode`` that
was used for feature extraction.

.. note::
    All sequence models are TGMJ models; however, if you have done feature extraction using ``mode='slow'`` then
    you should specify ``mode='slow'`` for training the sequence model as well. This is because the default sequence model
    checkpoints for each ``mode`` were trained with features extracted based on that ``mode``.

You can also specify a checkpoint path for training the sequence model if you have previously trained the sequence model
and want to start training from those weights instead. In this case, the ``mode`` argument will be ignored as a TGMJ model
will be constructed regardless. At this point, it is up to you as the user to know that the features extracted prior
to training were done with a given ``mode``.

.. code-block:: python

    # run feature extraction with mode='slow'
    for ix, row in df.iterrows():
        row.feature_extractor.infer(mode='slow')

    # train sequence model from pre-trained checkpoint, mode='slow'
    # save model checkpoint to certain output location
    sequence_out = '/path/to/sequence/model/outputs/'

    df.sequence.train(mode='slow', model_out=sequence_out)

    # train second dataframe from sequence model checkpoint from prior training
    # checkpoint will be located in previous specified output location from above
    sequence_checkpoint = '/path/to/sequence/checkpoint.ckpt'

    # mode argument will get ignored
    df2.sequence.train(model_in=sequence_checkpoint)

Using Your Own Model Checkpoints - Inference
--------------------------------------------

You can also run inference using non-default model checkpoints. The two main components of inferring behavior is
feature extraction and sequence model inference.

If you simply want to run inference using the default pre-trained model checkpoints you can use the following:

.. code-block:: python

    # run inference using mode='slow'
    for ix, row in df.iterrows():
        row.behavior.infer(mode='slow')

This will run feature extraction and sequence inference both for you.

If you want to use your own model checkpoints, you will need to run feature extraction and sequence inference separately.

**Feature Extraction**

.. code-block:: python

    # feature extraction using certain flow generator and feature extractor checkpoint
    feature_checkpoint = '/path/to/feature/extractor.ckpt'
    flow_checkpoint = '/path/to/flow/generator.ckpt'

    # run feature extraction for each row in the dataframe
    for ix, row in df.iterrows():
        row.feature_extractor.infer(flow_model_in=flow_checkpoint, feature_model_in=feature_checkpoint, mode=<mode>)

.. note::
    As mentioned in the section on training above, in order to properly reconstruct the models the model checkpoints
    must be to models that correspond to the flow generator and feature extractor models for a given ``mode`` argument.

**Sequence Inference**

Once you have run feature extraction, you may want to also use your own sequence model checkpoint for inference to get
the best results.

.. code-block:: python

    # sequence inference using a certain model checkpoint
    sequence_checkpoint = '/path/to/sequence/checkpoint.ckpt'

    # run sequence inference for each row in the dataframe
    for ix, row in df.iterrows():
        row.sequence.infer(model_in=sequence_checkpoint)

Similar to training the sequence model, the ``mode`` argument will be ignored when using your own checkpoint.