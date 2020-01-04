# Implementation an RNN time-embedding layer from "Time-Dependent Representation for Neural Event Sequence Prediction" paper

This is an implementation of the best performing approach to embedding time into RNNs. This is usefull for cases there your stream of RNNs steps contain timestep information,
and those timesteps are non-equally spaced. This approach performs better than just adding timestep as a single feature.

You can read detailed explanation in [my blog post](https://fridayexperiment.com/how-to-encode-time-property-in-recurrent-neutral-networks/)

References:

1. [paper - Time-Dependent Representation for Neural Event Sequence Prediction](https://arxiv.org/abs/1708.00065)