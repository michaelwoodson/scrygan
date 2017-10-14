# A TensorFlow project to process raw audio with LSTMs.

High level description available in [the blog post](https://michaelwoodson.github.io/listening-for-orcas-with-an-lstm/).

Details about parameters in [default_params.yaml](https://github.com/michaelwoodson/scrynet/blob/master/default_params.yaml).

Workflow is `train.py` (train the model) -> `scan.py` (record raw outputs) -> `tsunami.py` (create wav files to view outputs with Audacity).

This project used the [Tensorflow WaveNet Implementation](https://github.com/ibab/tensorflow-wavenet) as scaffolding.
