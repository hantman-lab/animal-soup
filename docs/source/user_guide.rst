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


Customization/Expansion
=======================

