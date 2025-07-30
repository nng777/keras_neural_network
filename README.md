# Lesson 1: Your First Neural Network with Keras

Welcome to AI Engineering! Today, we're going to build our first simple "brain"—a neural network. This guide will walk you through it step-by-step, making everything as clear as possible.

---

### The Big Idea: What Are We Doing?

Imagine you want to teach a computer to recognize a pattern—for example, to tell the difference between a picture of a cat and a picture of a dog. A **Neural Network** is a way of teaching a computer to do this. It's a system inspired by the human brain.

When a neural network has several layers of learning, we call it **Deep Learning**. Think of it like this:

-   **Neural Network:** A team of workers trying to solve a problem.
-   **Deep Learning:** A *big* team of workers organized into specialized departments (the layers). More departments mean they can solve more complex problems.

### The Smallest Worker: The Artificial Neuron

Our network is built from tiny workers called **neurons** (or **Perceptrons**). A single neuron is like one person making a simple decision.

Here’s how it works:
1.  **It gets information (Inputs):** These are the data points you give it.
2.  **It weighs the information (Weights):** It decides how important each piece of input is. A smart neuron learns to give more weight to more useful information.
3.  **It has a starting opinion (Bias):** This is like a thumb on the scale. It helps the neuron make better decisions by providing a baseline.
4.  **It makes a final call (Activation Function):** After looking at the inputs and weights, it makes a decision, like "yes" or "no."

### Our Toolkit: What is Keras?

**Keras** is like a Lego set for building neural networks. Instead of writing complex code from scratch, Keras gives us ready-made blocks (like `Layers`, `Optimizers`, etc.) that we can snap together easily.

--- 

### The 4 Steps to Building a Neural Network

Building a "brain" in Keras always follows the same simple workflow. Let's think of it as setting up a factory assembly line.

#### Step 1: Design the Assembly Line (Define the Model)

First, we design our assembly line. We use the `Sequential` model, which is a simple, straight line of workstations (layers).

-   **Layers:** These are the workstations on our assembly line.
    -   **Input Layer:** Where the raw materials (our data) enter the factory.
    -   **Hidden Layer (`Dense`):** The main workstations where the workers (neurons) process the information. `Dense` means every worker in a station talks to every worker in the next station.
    -   **Output Layer:** The final station, where the finished product (the prediction) comes out.

```python
# This is what it looks like in code:
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=[2]), # A hidden layer with 16 workers
    layers.Dense(1, activation='sigmoid')                # The final output station
])
```

#### Step 2: Give the Workers Their Instructions (Compile the Model)

Next, we give our workers their instructions for how to do their job well.

-   **The Goal (Loss Function):** We need to tell the factory what a "good product" looks like. The loss function measures how wrong the factory's predictions are. The goal is to get the **loss as low as possible**. For yes/no problems, we use `binary_crossentropy`.
-   **The Teacher (Optimizer):** The optimizer is like a teacher that tells the workers how to get better. It adjusts their weights to reduce the loss. `adam` is a very smart and effective teacher.
-   **The Report Card (Metrics):** We need a way to grade our factory. We use `accuracy` to see what percentage of the time our network makes the correct prediction.

```python
# Giving the instructions in code:
model.compile(
    optimizer='adam',             # The smart teacher
    loss='binary_crossentropy',   # The goal: minimize error
    metrics=['accuracy']          # The report card
)
```

#### Step 3: Start the Factory (Train the Model)

Now it's time to turn on the machines and start learning! We feed our data to the model using the `.fit()` command.

-   **Epochs:** An epoch is one full run through the entire training dataset. If we set `epochs=50`, it means our factory will practice on all the data 50 times to get better.
-   **Batch Size:** Instead of learning from one piece of data at a time, the model learns from a small group (a "batch") of them. This is more efficient.

```python
# Training in code:
# model.fit(X_train, y_train, epochs=50, batch_size=32)
```

#### Step 4: Perform Quality Control (Evaluate the Model)

Finally, we test our model on data it has **never seen before**. This is the true test of how well it learned.

```python
# Evaluating in code:
# loss, accuracy = model.evaluate(X_test, y_test)
```

And that's it! You now understand the core concepts and workflow for building your very first neural network.