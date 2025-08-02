import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# --- 1. Data Preparation ---
# We will generate a simple, linearly separable dataset for this example.
# X will have 2 features, and y will be a binary class (0 or 1).
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the data into a training set and a testing set.
# The model will learn from the training set and be evaluated on the testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}")
print("---")


# --- 2. Build the Neural Network Model ---
# We use the Sequential API, which is a simple stack of layers.
model = keras.Sequential([
    # The input layer is implicitly defined by `input_shape`.
    # Our first hidden layer has 16 neurons and uses the 'relu' activation function.
    # `input_shape=[2]` because our data has 2 features.
    layers.Dense(16, activation='relu', input_shape=[2], name='hidden_layer_1'),

    # TASK 1: A second hidden layer with 8 neurons using 'relu' activation for added model capacity.
    layers.Dense(8, activation='relu', name='hidden_layer_2'),

    # The output layer has 1 neuron because this is a binary classification problem.
    # We use the 'sigmoid' activation function to get a probability output between 0 and 1.
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Display a summary of the model's architecture.
print("Model Architecture:")
model.summary()
print("---")


# --- 3. Compile the Model ---
# Here, we configure the model for training.
# - Optimizer: 'adam' is an efficient and popular choice.
# - Loss Function: 'binary_crossentropy' is used for two-class (binary) problems.
# - Metrics: We want to monitor the 'accuracy' during training.
model.compile(
    optimizer='sgd', # TASK 2: optimizer='adam' to sgd model
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# --- 4. Train the Model ---
# We "fit" the model to our training data.
# - epochs: The number of times the model will cycle through the entire training dataset.
# - batch_size: The number of samples processed before the model's weights are updated.
# - verbose: 1 shows a progress bar, 0 is silent.
print("Starting model training...")
history = model.fit(
    X_train,
    y_train,
    epochs=100, # TASK 3: increase step epochs=50 to 100
    batch_size=32,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)
print("---")


# --- 5. Evaluate the Model ---
# We check the model's performance on the test data it has never seen before.
print("Evaluating model on test data...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print("---")


# --- 6. Make a Prediction ---
# Let's see how the model predicts a new, unseen data point.
# The output should be close to 1.
new_data_point = np.array([[2, 2]])
prediction_probability = model.predict(new_data_point)[0][0]
predicted_class = (prediction_probability > 0.5).astype("int32")

print(f"Predicting for new data point: {new_data_point}")
print(f"Predicted Probability: {prediction_probability:.4f}")
print(f"Predicted Class (0 or 1): {predicted_class}")
