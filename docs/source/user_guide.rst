User Guide
==========

The demo notebook is the best place to start: https://github.com/hantman-lab/animal-soup/blob/main/notebooks/demo_notebook.ipynb

This guide provides some more details on the API and concepts for using ``animal-soup``.

Animal-soup interfaces is a collection of "pandas extensions" -- functions that operate on `pandas DataFrames <https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe>`_. This enables you to create a "psuedo-database" of your behavioral data. No database setup or experience is required, it operates purely on ``pandas`` and standard file systems.

Since this framework uses ``pandas`` extensions, you should be relatively comfortable with basic pandas operations. If you're familiar with ``numpy`` then ``pandas`` will be easy, here's a quick start guide from the pandas docs: https://pandas.pydata.org/docs/user_guide/10min.html

Accessors and Extensions
========================

There are 4 *accessors* that the ``animal-soup`` API provides, ``behavior``, ``flow_generator``, ``feature_extractor`` and ``sequence``. These allow you to perform operations on a ``pandas.DataFrame``.

Each row in an ``animal-soup`` dataframe corresponds to a **single trial**.

A **single trial** is the combination of:

* animal_id
* session_id
* trial_id
* vid_path
* ethograms
* exp_type
* model_params
* notes

**Examples:**

Some common ``behavior`` extensions are:

* ``behavior.add_item()`` - adds a single trial, a single session, or all sessions for an animal to the dataframe
* ``behavior.remove_item()`` - removes a single trial, a single session, or all sessions for an animal from the dataframe
* ``behavior.view()`` - creates a container for viewing your behavioral data

Some flow generator extensions:

* ``flow_generator.train()``

Some feature extractor extensions:

Some sequence extensions: