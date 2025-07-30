# Homework Assignment: Experiment with the Model

Your assignment is to modify the `app.py` script to see how changes to the model's architecture and training process affect its performance.

### Your Task

Make the following three changes to `app.py`:

1.  **Add a Layer:** Add a second `Dense` hidden layer to the model. It should have **8 neurons** and use the `relu` activation function. Place it between the existing hidden layer and the output layer.

2.  **Change the Optimizer:** In the `model.compile()` step, change the optimizer from `'adam'` to `'sgd'` (Stochastic Gradient Descent).

3.  **Increase Training Time:** In the `model.fit()` step, increase the number of `epochs` from `50` to `100`.

### To Do

1.  Modify the `app.py` file with the changes described above.
2.  Run the modified script: `python app.py`.
3.  Observe the final "Test Accuracy" in the output. Did it improve, get worse, or stay about the same? Be prepared to discuss why you think the performance changed.
