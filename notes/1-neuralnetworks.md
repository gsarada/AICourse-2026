## AI Landscape

[AI](AI_Landscape.png)

## Artificial Neural Networks

- Intro: Neural Networks (NNs) are networks of neurons, as found in real brains. Artificial neurons are crude approximations of the neurons found in brains. They may be physical devices, or purely mathematical/software constructs.
- Definition: Artificial Neural Networks are networks of Artificial Neurons. It is a parallel computational system consisting of many simple processing elements connected in a sepecific way to perform a particular task.
- Neural networks consist of interconnected layers (input, hidden, output) that process data using weighted connections and activation functions. Key aspects include structure (layers/nodes), architecture types (feedforward, CNN, RNN), training processes (backpropagation), and applications like image recognition or predictive analytics.
- These networks are crucial for diverse applications, including computer vision, natural language processing, and predictive modeling.

### Key Aspects of Neural Networks:

1. Structure and Components:
  - Layers: Input layer (receives data), hidden layers (process information), and output layer (produces results).
  - Nodes/Neurons: Artificial neurons receive input, apply weights and biases, and pass the data through an activation function.
  - Weights and Biases: Parameters adjusted during training to minimize error, allowing the network to learn.
2. Architecture Types:
  - Feedforward Neural Networks (FNN): Data moves in one direction, from input to output.
  - Convolutional Neural Networks (CNN): Designed for image processing and spatial data.
  - Recurrent Neural Networks (RNN): Handle sequential data by allowing information to loop.
  - Generative Adversarial Networks (GANs): Used for creating new data instances.
3. Training and Learning:
  - Backpropagation: The fundamental algorithm for training, which adjusts weights based on error.
  - Activation Functions: Non-linear functions (e.g., ReLU, Sigmoid) that allow networks to learn complex patterns.
4. Performance and Challenges:
  - Overfitting: When a network learns training data too well, failing to generalize to new data; mitigated by dropout or regularization.
  - Interpretability: The "black box" nature of deep learning, making it hard to explain how decisions are made.
[Details](2-nnbasics.md)


## Neural Networks Classification/Evolution

[ Umbrella: Artificial Neural Network (ANN) ]
       │
       └───► [ Deep Neural Network (DNN) ]
                    │
                    ├───► [ Recurrent Neural Network (RNN / LSTM) ]
                    │
                    └───► [ Modern Attention Networks (Transformers) ]

1. Classical and Feed-Forward Networks: These networks process information in one single direction, from the input layer to the output layer, without any internal loops or memory pathways.
  - Perceptron: The simplest, original form of a neural network consisting of a single neuron. It takes binary inputs, applies weights, and uses a step activation function to produce a binary output.
  - Multilayer Perceptron (MLP): The baseline Artificial Neural Network (ANN). It features an input layer, one or more hidden layers, and an output layer where every neuron in one layer connects to every single neuron in the next (fully connected). They are primarily used for structured, tabular data like spreadsheets or database predictions.
2. [Spatial and Pattern Recognition Networks](4-spatialnetworks.md): Built specifically to process grid-like, multi-dimensional structures where localized context and spatial relationships are critical.
  - Convolutional Neural Networks (CNN / DCNN / U-Net): Instead of looking at an entire image at once, CNNs use small moving windows called filters to scan across data. This allows them to capture spatial features (edges, textures, shapes) efficiently, making them the standard choice for computer vision, image classification, and medical scans. U-Net utilizes this structure symmetrically with skip connections for precise image segmentation.
3. [Sequential and Temporal Networks](5-temporalnetworks.md): Designed for data where the order of information matters, such as sentences, audio recordings, or stock prices over time.
  - Recurrent Neural Networks (RNN): These networks introduce feedback loops, allowing them to pass past information forward to process the next step in a sequence.
  - Long Short-Term Memory (LSTM): An advanced variant of an RNN that uses internal gates (forget, input, and output gates) to preserve long-term memory dependencies, solving the problem of recurrent networks forgetting early text or data inputs.
4. [Modern Attention-Based Networks (Transformers)](6-transformers.md): The current state-of-the-art framework that processes entire chunks of data simultaneously rather than step-by-step. They utilize Self-Attention to calculate mathematical relationships between all data points (tokens or image patches) concurrently, forming the backbone of modern LLMs and vision models.
  [Transformer Lineages](7-transformerlineage.md)
  - Lineage A (Encoder-Only): Used for understanding context, classification, and feature extraction. Examples include BERT (text) and ViT (Vision Transformer, which processes images as flattened patch sequences).
  - Lineage B (Decoder-Only): Used for autoregressive text or data generation, predicting the next token in a sequence. Examples include the GPT family.
  - Lineage C (Encoder-Decoder): Used for sequence-to-sequence translation and summarization tasks. Examples include BART, PEGASUS, and T5.
5. [Unsupervised and Generative Frameworks](8-generativeframeworks.md): Architectures designed to compress data, map latent spaces, or synthesize entirely new content from scratch.
  - Generative Adversarial Networks (GAN): A dual-network framework where a Generator tries to create realistic fake data, while a Discriminator tries to spot the fakes. They train by competing against each other.
  - Autoencoders (AE / VAE): Hourglass-shaped networks that compress input data down into a tiny bottleneck layer (latent space) and then attempt to perfectly reconstruct the original data from that compression. Heavily utilized for noise reduction and anomaly detection.
  - Diffusion Models (Stable Diffusion): A generative mathematical framework that creates clean images by iteratively removing random Gaussian noise from a canvas step-by-step. They use independent neural backbones—traditionally a CNN (U-Net), but increasingly an Encoder-Only Transformer (DiT / Diffusion Transformer).
  - Compound Generative Systems (DALL-E): Production-ready AI software ecosystems that do not rely on a single architecture. Instead, they hybrids that weave multiple components together—combining Encoder-Only Vision Transformers (like CLIP) for text-image alignment, large language decoders for prompt comprehension, and diffusion models for final image synthesis.
6. [Multi-modal networks](9-multimodalnetworks.md): These are artificial intelligence networks engineered to process, understand, and synthesize multiple distinct types of data—such as text, images, audio, video, and sensory data—simultaneously within a single unified network architecture


## Neural Networks Use Cases
1. Natural Language Processing [(NLP)](3-nlp.md) and Speech
2. Computer Vision and Image Processing
3. Predictive Analysis and Tabular Data
4. Cyber Security and Anomaly Detection

