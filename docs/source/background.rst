Background
**********

``animal-soup`` uses a series of convolutional neural networks (CNNs) in order to perform automated animal behavior classification of the Hantman Lab reach-to-grab task.

.. note::
    The ``animal-soup`` architecture was adopted from DeepEthogram.
    You can find their eLife paper `here <https://elifesciences.org/articles/63377>`_ and their GitHub repo `here <https://github.com/jbohnslav/deepethogram>`_.

There are three main model components: a flow generator, feature extractor, and sequence model.

Flow Generator
==============

The flow generator is a CNN that calculates optic flow. It takes a given window size (default 11) and creates “clips” to generate optic flow features.
Optic flow features summarize the motion across frames and can be used in determining the behavior at a given time point of a trial.

There are three different flow generator models that can be used: TinyMotionNet3D, MotionNet, and TinyMotionNet. Their
corresponding modes are listed below:

+--------+-----------------+
| mode   | model            |
+========+=================+
| fast   | TinyMotionNet   |
+--------+-----------------+
| medium | MotionNet       |
+--------+-----------------+
| slow   | TinyMotionNet3D |
+--------+-----------------+

The primary difference between the models is the number of layers. As a result, there is
higher accuracy but at the consequence of speed.

Feature Extractor
=================

The feature extractor is a two-stream fused model that extracts the relevant features in each
frame.

The model consists of a flow and spatial classifier. The flow classifier takes in optic flow features
from a flow generator and the spatial classifier takes in individual raw frames. The results of these two
classifiers is a lower dimensional representation of the features in a given trial.

The type of flow and spatial classifiers constructed are based on the ResNet models listed below:

+--------+---------------+
| mode   | feature model |
+========+===============+
| slow   | ResNet3D-34   |
+--------+---------------+
| medium | ResNet50      |
+--------+---------------+
| fast   | ResNet18      |
+--------+---------------+

Again, the primary difference between the models is the number of layers which introduces the
same accuracy versus speed dilemma as above.

Sequence Model
==============

The sequence model is a TGMJ model. This is a type of Temporal Gaussian Mixture model that is used
for activity detection across a series of sequences from a trial. This model allows for long-term
learning over a temporal period.

The model takes in the spatial and flow features extracted by the feature extractor and seeks to
give the probabilities of a given behavior occurring at each time point. These probabilities can
be used to then create a binary matrix (number of behaviors, number of time points) called an ethogram
that represents the behavioral classification of a given trial.