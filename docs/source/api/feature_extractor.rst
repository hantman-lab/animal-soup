.. _api_extensions_feature_extractor:

Feature Extractor
*****************

Extensions that are used for training and running inference with the sequence model.

**Accessor:** ``feature_extractor``

DataFrame Extensions
====================

These can be called on the DataFrame

.. autoclass:: animal_soup.FeatureExtractorDataframeExtension
    :members:

Series Extensions
=================

These can be called on an individual Series, or rows, of the DataFrame

.. autoclass:: animal_soup.FeatureExtractorSeriesExtensions
    :members:
