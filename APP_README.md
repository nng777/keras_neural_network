# Practical Keras Example: `app.py` - Technical Guide

This guide provides a technical walkthrough of the `app.py` script, explaining how to run it and what each part of the code does from a machine learning perspective.

### Purpose

The `app.py` script is a practical, hands-on demonstration of a complete, albeit simple, machine learning workflow. It covers:
1.  **Data Generation:** Creating a synthetic, linearly separable dataset suitable for a basic classification task.
2.  **Model Definition:** Building a neural network with the Keras `Sequential` API.
3.  **Model Compilation:** Configuring the learning process with an optimizer and loss function.
4.  **Model Training:** Fitting the model to the data to learn the underlying patterns.
5.  **Model Evaluation:** Measuring the model's performance on unseen data.
6.  **Inference:** Using the trained model to make a prediction on a new data point.

---

### 1. Setup and Installation

To execute the script, you need to install the core libraries. It is best practice to use a Python virtual environment to avoid conflicts with system-wide packages.

Open your terminal and run the following command:

```bash
pip install tensorflow scikit-learn numpy
```

-   **`tensorflow`:** The core deep learning framework. Keras is a high-level API that is officially part of TensorFlow, making it easy to build and train models.
-   **`scikit-learn`:** A powerful machine learning library. We use its `make_classification` function to generate a well-behaved, synthetic dataset for our classification task and `train_test_split` to partition our data.
-   **`numpy`:** The fundamental package for numerical operations in Python, used for handling the data arrays.

---

### 2. Running the App

After installation, execute the script from your terminal:

```bash
python app.py
```

---

### 3. Technical Breakdown of the Script

#### a. Data Generation and Splitting
```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
-   `make_classification` creates a dataset of `(X, y)` pairs. We configure it to have 1000 samples, 2 predictive (`informative`) features, and 2 distinct classes (since `n_clusters_per_class=1` and the default `n_classes=2`). This creates two clear clusters of data points, making it an ideal introductory problem.
-   `train_test_split` partitions the data. The model learns *only* from `X_train` and `y_train` (80% of the data). The `X_test` and `y_test` (20%) are held back as unseen data to provide an unbiased evaluation of the model's performance.

#### b. Model Architecture
```python
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=[2]),
    layers.Dense(1, activation='sigmoid')
])
```
-   `keras.Sequential`: We use the simplest model type, a linear stack of layers.
-   `layers.Dense(16, ...)`: This is our hidden layer with 16 neurons. The `Dense` name means every neuron in this layer is connected to every input feature.
    -   `activation='relu'`: The Rectified Linear Unit is the standard activation for hidden layers. It introduces non-linearity, allowing the model to learn more complex relationships than a simple linear model.
    -   `input_shape=[2]`: This tells the model to expect input data with 2 features. This is only required on the first layer.
-   `layers.Dense(1, ...)`: This is the output layer.
    -   It has a single neuron because we are performing binary classification (the output will be a single number).
    -   `activation='sigmoid'`: The sigmoid function squashes the output to a value between 0 and 1. This is crucial as it allows us to interpret the output as a probability.

#### c. Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
-   `optimizer='adam'`: Adam (Adaptive Moment Estimation) is a highly effective and commonly used optimization algorithm. It adapts the learning rate during training, which often leads to faster convergence.
-   `loss='binary_crossentropy'`: This loss function is the mathematical choice for binary (two-class) classification problems. It measures the difference between the predicted probability (from the sigmoid function) and the true binary label (0 or 1).

#### d. Training and Evaluation
```python
model.fit(X_train, y_train, epochs=50, ...)
loss, accuracy = model.evaluate(X_test, y_test)
```
-   `model.fit`: This command starts the training loop. The optimizer (`adam`) will try to adjust the model's weights to minimize the `binary_crossentropy` loss.
    -   `epochs=50`: The model will iterate over the entire training dataset 50 times.
    -   `batch_size=32`: The model processes the data in mini-batches of 32 samples at a time. After each batch, it calculates the loss and updates the weights.
-   `model.evaluate`: After training, this command assesses the final model on the held-back test set, giving us a clear measure of its `loss` and `accuracy` on unseen data.

#### e. Inference
```python
prediction_probability = model.predict(new_data_point)[0][0]
predicted_class = (prediction_probability > 0.5).astype("int32")
```
-   `model.predict()` returns the raw output from the final sigmoid layer—a probability.
-   To get a concrete class (0 or 1), we apply a threshold. The standard threshold is 0.5. If the predicted probability is greater than 0.5, we classify it as 1; otherwise, we classify it as 0.