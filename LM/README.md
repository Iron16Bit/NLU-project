# NLU Project - 2025

## Lab 4 Mandatory Exercises

### Part 1-A

In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment. For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
One of the important tasks of training a neural network is hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular the learning rate). These are two links to the state-of-the-art papers which use vanilla RNN.

Mandatory requirements: For the following experiments the perplexity must be below 250 (PPL < 250).

- Replace RNN with a Long-Short Term Memory (LSTM) network
- Add two dropout layers:
    - one cle,
    - one before the last linear layer
- Replace SGD with AdamW

### Part 1-B

Mandatory requirements: For the following experiments the perplexity must be below 250 (PPL < 250) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the LM_RNN in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:

- Weight Tying
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD

## Lab 5 Mandatory Exercises