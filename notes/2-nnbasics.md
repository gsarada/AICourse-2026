## Training Process
The training process within a single epoch follows a strict, sequential pipeline to process data, evaluate performance, and update model parameters.
1. Batch partitioning: The entire training dataset is split into smaller subsets called mini-batches. The algorithm loops through these mini-batches one by one until all training samples have been processed exactly once, completing the epoch. Data is usually shuffled at the start of each epoch to prevent the model from learning the order of the samples.
2. The training loop: For every mini-batch inside the epoch, the network executes the following four sequential operations:
  - Forward pass: The mini-batch data is fed into the network. It multiplies inputs by weights, adds biases, and passes results through activation functions layer by layer to generate a prediction.
  - Loss computation: A loss function compares the network's final predictions against the actual ground-truth labels. It calculates a single scalar value representing the prediction error for that batch.
  - Backward pass: The network calculates the gradient of the loss function with respect to every weight and bias using the calculus chain rule. This maps how much each specific parameter contributed to the error.
  - Parameter update: The optimization algorithm (like SGD or Adam) uses the calculated gradients to modify the internal weights. This shifts the parameters in the direction that minimizes the overall error.
3. Metric Evaluation: Training loss and accuracy are aggregated for entire epoch. The validation set if provided will run through forward pass to determine the metrics on unseen data.

## Common Types of Neural Network Layers
1. Input Layer: The initial layer that receives raw input data, with neurons equal to the number of input features.
2. Dense / Fully Connected Layer: Connects every neuron in one layer to every neuron in the next, often used in final classification tasks.
3. Convolutional Layer (CONV): The core of Convolutional Neural Networks (CNNs), it uses filters to detect spatial features such as edges and textures in data like images. Multiple filters in a convolutional layer capture different patterns and structures from the input.
4. Pooling Layer (POOL): Reduces the dimensionality of feature maps (e.g., Max Pooling) to reduce computational load and control overfitting.
5. Recurrent Layer (RNN): Processes sequential data (like text or time series) by maintaining a hidden state that carries information from previous steps.
6. Normalization Layer (BN): Stabilizes training by normalizing the inputs to each activation function, often used in deep networks.
7. Dropout Layer: A regularization layer that randomly sets input units to 0 during training to prevent overfitting.
8. Activation Layer (ReLU, Sigmoid, Tanh): Introduces non-linear transformations into the network, enabling it to learn complex, non-linear relationships.
9. Deconvolutional Layer (Deconv): Performs the reverse of convolution, used for upsampling data in tasks like image segmentation.
10. Output Layer: The final layer that produces the network's prediction or classification.

## What is Layer weight:
Layer weights in a neural network are trainable numerical parameters associated with connections between neurons,
determining the strength, influence, and importance of input signals. They multiply incoming data to dictate how
much each feature contributes to the next layer's output. These weights are adjusted during training to optimize accuracy.
Key Aspects of Layer Weights:
1. Role in Learning: Weights are initially random and optimized during backpropagation to minimize the error between predicted and actual outputs.
2. Significance: A high weight value indicates a strong, important connection, while a weight near zero suggests the input has little influence on the output.
3. Directionality: Positive weights increase the activation of the next neuron, while negative weights decrease it, acting as inhibitory connections.
4. Computation: In a layer, input values are multiplied by corresponding weights, summed together, and often added to a bias term before passing through an activation function.
5. Structure: They are organized in matrices (often denoted as coefs_ in frameworks like scikit-learn) that connect neurons in one layer to those in the next.
Effectively, weights represent the learned "knowledge" of the network, enabling it to recognize patterns and make predictions.

## What is Layer Bias:
Layer bias in a neural network is a learnable, constant parameter added to the weighted sum of inputs for neurons within a layer, serving as an offset (b) to control the activation threshold.
It acts like the intercept in a linear equation (y=mx+c), allowing the network to shift activation functions (e.g., ReLU, Sigmoid) up or down, independent of input values, which improves flexibility and learning. 
Key Aspects of Layer Bias:
1. Definition & Function: Bias is a trainable parameter, initialized, and updated during backpropagation alongside weights.It ensures that even if all input features are zero, a neuron can still output a non-zero value and propagate information.
2. Mathematical Representation: The output of a neuron is calculated as y=f(\sum (w_{i}\times x_{i})+b), where w represents weights, x is input, and b is the bias.
3. Role in Learning: Without bias, a neuron might fail to activate if the weighted sum is always zero or negative (e.g., with ReLU). It allows the model to better fit the data, particularly for patterns that do not pass through the origin (0,0).
4. Difference from Weights: While weights determine the strength of the connection between neurons, the bias controls the activation threshold (sensitivity) of the neuron itself.
5. Implementation: In deep learning frameworks like TensorFlow or PyTorch, biases are typically included by default in linear layers.
In summary, layer bias provides the necessary freedom to shift the decision boundary, enabling the network to learn more complex, accurate functions.

## What is an activation function:
An activation function in a neural network is a mathematical function applied to a neuron's output, introducing non-linearity to enable the network to learn complex patterns beyond simple linear relationships, essentially deciding if and how a neuron should "fire" or pass information to the next layer, with common examples being ReLU, Sigmoid, and Tanh.
Without it, a deep network acts like a simple linear model, limiting its power for real-world tasks.
1. Key Functions & Roles: 
  - Introduce Non-Linearity: This is the most crucial role, allowing neural networks to model complex, non-linear data (like images, speech) instead of just straight lines.
  - Decision Making: It acts like a switch, determining if a neuron's input is significant enough to activate and pass on.
  - Output Transformation: It transforms the weighted sum of inputs and bias into a specific output range, which helps stabilize training.
  - Gradient Propagation: Its derivative is vital during backpropagation, influencing how weights are updated during training.
2. Common Types of Activation Functions: 
  - ReLU (Rectified Linear Unit): max(0, x). Popular for hidden layers, it's computationally efficient.
  - Sigmoid (Logistic): 1 / (1 + exp(-x)). Squashes output between 0 and 1, useful for binary classification.
  - Tanh (Hyperbolic Tangent): Squashes output between -1 and 1, often preferred over sigmoid in hidden layers.
  - Softmax: Used in the output layer for multi-class classification, converting scores into probabilities.
  - Leaky ReLU/PReLU: Variants of ReLU that allow a small, non-zero gradient for negative inputs to prevent "dying neurons".
3. Why They Matter: By adding these non-linear transformations, activation functions allow deep neural networks to build complex decision boundaries and represent intricate relationships in data, making them powerful tools for AI.

## What is back propagation:
Backpropagation (backward propagation of errors) is the fundamental algorithm for training neural networks, enabling them to learn from data by calculating gradients of a loss function and updating network weights. It works backward from output to input nodes, using the chain rule of calculus to efficiently compute how much each weight and bias contributes to the error, allowing for precise weight adjustments to minimize it.
1. Key Aspects of Backpropagation: 
  - Goal: To minimize the error (loss) of a network's prediction compared to the actual label by updating weights.
  - Process: It computes the gradient (derivative) of the loss function with respect to each parameter in the network.
  - Efficiency: Instead of calculating the gradient for each weight separately, it uses the chain rule to compute gradients from the last layer to the first (backward), avoiding redundant calculations.
2. Workflow:
  - Forward Pass: Input data goes through the network to produce an output.
  - Loss Calculation: The difference between the prediction and actual output is measured.
  - Backward Pass: The error is propagated backward, calculating gradients for each weight.
  - Weight Update: An optimizer (like gradient descent) uses these gradients to update the weights.
3. Why it Matters: Without backpropagation, training deep neural networks would be computationally infeasible due to the millions of parameters involved. It provides the "correction" step that allows neural networks to learn, making it essential for deep learning applications like image recognition, natural language processing, and speech recognition.

## What is compiling a model:
Compiling a model in a neural network is the process of configuring the model for training and preparing it for efficient execution on the target hardware. This step transforms the network's abstract definition into a highly efficient series of operations (often matrix transforms) tailored for the CPU or GPU, before the actual learning process begins.
Key Purposes of Compilation
1. Configuring the Learning Process: The primary function of compilation is to define the essential components and algorithms the model will use to learn from data. This includes specifying:
  - Optimizer: The algorithm that determines how the model's weights are updated during training to minimize the error. Common choices include 'adam', 'sgd' (Stochastic Gradient Descent), and 'rmsprop'. [https://keras.io/api/optimizers/](https://keras.io/api/optimizers/)
  - Loss Function: The mathematical function that quantifies the difference between the model's predictions and the actual target values. The goal of training is to minimize this value. Common loss functions include 'mean_squared_error' (for regression) and 'binary_crossentropy' or 'categorical_crossentropy' (for classification). [https://keras.io/api/losses/](https://keras.io/api/losses/)
  - Metrics: Optional metrics used to evaluate the model's performance and report during training (e.g., 'accuracy' for classification problems). More at [https://keras.io/api/metrics/](https://keras.io/api/metrics/)
2. Optimizing for Hardware: Compilation involves an efficiency step where the framework (like TensorFlow or PyTorch) optimizes the network's structure for faster execution on available hardware. This can involve:
  - Backend Selection: The software backend automatically chooses the best way to represent the network for efficient training and prediction runs.
  - Performance Enhancements: Compilation can remove overhead and apply optimizations (such as JIT compilation in PyTorch 2.0) that improve inference speed and efficiency, making the model suitable for deployment in various applications and devices.

## Possible ways to improve model
1. Tuning Hyperparameters: A good hyperparameter to start with is the learning rate for the Adam optimizer. What about the batch size and number of epochs? How does learning rate schedule improve the model
2. Network Depth: What happens if we remove or add more fully-connected layers? How does that affect training and/or the model’s final performance? How about neurons per layer?
3. Activations: What if we use an activation other than ReLU, e.g. leakyRelu, sigmoid?
4. Dropout: What if we tried adding Dropout layers, which are known to prevent overfitting?
5. Validation: We can also use the testing dataset for validation during training. Keras will evaluate the model on the validation set at the end of each epoch and report the loss and any metrics we asked for. This allows us to monitor our model’s progress over time during training, which can be useful to identify overfitting and even support early stopping.

## Parameters to evaluate model performance
1. Loss/ Accuracy: When the training loss continues to decrease but the validation loss starts to increase, or when the training accuracy goes up and validation accuracy goes down, it is a classic sign of overfitting. This means the model is learning the specifics and noise within the training data too well, to the detriment of its ability to generalize to new, unseen data.
2. Precision (Positive Predictive Value): Answers "Of all predicted positives, how many were actually positive?". High precision is vital when the cost of a false positive is high (e.g., spam detection).
3. Recall (Sensitivity/True Positive Rate): Answers "Of all actual positives, how many did we find?". High recall is critical when the cost of a false negative is high (e.g., cancer diagnosis, fraud detection).
4. F1 Score: A single metric that combines precision and recall. It is a better measure than accuracy when data is imbalanced, as it requires both to be high.
5. Receiver Operating Characteristic (ROC) Curve: A graph plotting a model's True Positive Rate (TPR, Y-axis) against its False Positive Rate (FPR, X-axis) at various classification thresholds.
  - True Positive Rate (TPR) / Sensitivity: (True Positives / (True Positives + False Negatives)) - How well it identifies actual positives.
  - False Positive Rate (FPR): (False Positives / (False Positives + True Negatives)) - The proportion of actual negatives incorrectly classified as positive.
6. Area Under the Curve (AUC): The area beneath the ROC curve, a single number summarizing the model's overall discriminative ability.
  - Interpretation:
  AUC = 1.0: Perfect classifier (100% separation).
  AUC = 0.5: No better than random guessing (model can't distinguish classes).
  AUC > 0.8: Generally considered clinically useful/good performance.

## Building ANN model structure
1. model = Sequential() -- Init
2. model.add(Dense(20, input_dim=8, activation='relu')) -- adding dense layer to the NN. 20 is no of neurons in that layer, input_dim 8 means it receives 8 features as input variables and the activation function is relu
3. model.add(Dense(...)) -- can add multiple dense layers with different number of neurons and activation function
4. model.add(Dense(..))
5. model.add(Dense(1, activation='sigmoid')) -- this is sample of final layer. 1 means it produces single output and sigmoid is the activation function which usually produces label data (0 or 1)

## Compiling, Training and evaluating ANN model
1. model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) -- compiling the model with various functions which help the model during training to achieve better results
2. model.fit(Xtr, ytr, epochs=300, batch_size=10) -- training the model by passing in features and labels. Epochs is the number of iterations the model will undergo training. Batch_size is the number of input samples taken for each run within an iteration
3. model.evaluate(Xtst, ytst, verbose=0) -- evaluating the model performance in terms of accuracy on test set
4. model.predict(Xtst) -- Predicting the labels for test sample
5. Use classification report and confusion matrix to evaluate the model performance metrics and update the layers if required

## Factors for Non determinism in model training
1. Software-level randomness
  - Weight initialization: Frameworks generate random starting values for weights using statistical distributions. Without an explicit constraint, a new random seed is used every time we hit run.
  - Data shuffling: To ensure the model learns generalized features rather than memorizing data order, mini-batches are randomly shuffled at the start of every epoch.
  - Stochastic regularization: Layers like Dropout randomly deactivate different subsets of neurons during every single forward pass, altering the mathematical path of execution.
2. Hardware-level randomness Even if we perfectly lock down all software seeds, training on a GPU remains non-deterministic. High-performance computing frameworks (like NVIDIA's cuDNN) execute millions of floating-point operations simultaneously out of order. Because computer rounding changes depending on calculation order, this introduces microscopic mathematical differences that compound over millions of backpropagation steps
  - To achieve 100% determinism, we can explicitly lock all software random seeds and force the GPU to use deterministic, sequential algorithms at the cost of training speed.
    - Set environment variables to force deterministic GPU algorithms
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    - Set all software random seeds simultaneously
    tf.keras.utils.set_random_seed(42)
