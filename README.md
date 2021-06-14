# Hidden Markov Model

## Introduction

Hidden Markov Model (HMM) is a statistical Markov model in which the system being modelled is assumed to be a Markov process Z with unobservable "hidden" states. HMM assumes that there is another process X whose behaviour "depends" on Z. The goal is to learn about Z by observing X.

The HMM is defined by 3 parameter matrices:
- Initial matrix
- Transition matrix
- Emission matrix

With a trained HMM, the following can be obtained:
- The most likely sequence of hidden states based on a given sequence of observations
- The likelihood of a given sequence of hidden states

## Methods

### Train
In this endpoint, the user assumes the hyper-parameter, which is number of hidden states. Using observed sequences, Expectation Maximization is performed to calculate the probabilities across HMM's 3 parameter matrices.

### Update
In this end point, the user already has an initial belief of the underlying HMM structure and its parameters. The user also has new observation sequences she wants to incoporate into the prior HMM parameters, formulate a new posterior belief on the parameters.

### Predict
In this end point, the user already has the most updated belief on the underlying HMM structure and its parameters, and wants to find out the most likely sequence of underlying hidden states for a given observation.

