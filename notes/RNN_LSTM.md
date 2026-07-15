## RNN
A Recurrent Neural Network (RNN) is the baseline neural network architecture designed to handle sequential data or time-series data.Unlike standard feed-forward networks (MLPs) that assume all inputs and outputs are completely independent of each other, an RNN possesses an internal looping mechanism. This allows information to persist, meaning the network can utilize past inputs to better understand current data points.

The core concept: Loops and hidden states. A standard network processes an input and yields an output in a single forward pass, instantly forgetting what it just saw. An RNN introduces a feedback loop, meaning the network passes its output back into itself.  

          ┌───┐
          ▼   │ (Looping Memory)
     [ RNN Cell ]
          ▲   │
          │   ▼
     [ Input ]  ──► [ Output ]


To visualize this easily, engineers mathematically unroll the network across time steps:


     Time Step 1:               Time Step 2:               Time Step 3:
     [ Input x0 ]               [ Input x1 ]               [ Input x2 ]
          │                          │                          │
          ▼                          ▼                          ▼
     [ RNN Cell ] ── Hidden State ─►[ RNN Cell ] ── Hidden State ─►[ RNN Cell ]
          │   (Memory 0)             │   (Memory 1)             │
          ▼                          ▼                          ▼
     [ Output h0 ]              [ Output h1 ]              [ Output h2 ]


At every individual time step (t), the network takes two separate inputs:
  - The current input (\(x_{t}\)): The immediate item in the sequence (e.g., the current word in a sentence or today's stock price).
  - The previous hidden state (\(h_{t-1}\)): The compressed mathematical memory summarizing everything the network has processed from step 0 up until now.

The RNN merges these two inputs, applies its weights, and outputs a new hidden state (\(h_{t}\)) that acts as the updated memory passed directly into the next time step.

## Why RNNs are essential for sequential data
Standard networks fail on sequential tasks because they lack contextual awareness and require rigid, fixed-size inputs. RNNs solve both challenges:
  * Contextual awareness: In the phrase "I am going to the bank to cash a check," the network cannot know if "bank" means a riverbank or a financial institution until it reads the end of the sentence. An RNN uses its memory to carry the context forward.
  * Variable input length: Because the same RNN cell is simply reused over and over again for every single item in a sequence, the input text or timeline can be 5 items long or 5,000 items long. Explanation - 
    - The fixed-size limitation of traditional networks: A standard feed-forward network (like a Multilayer Perceptron or a DCNN) requires an exact, unchangeable input size defined in its very first layer.If you design a dense layer with 784 input nodes, it can only accept vectors of exactly 784 numbers. If you try to feed it an image that is slightly larger or a text sentence with three extra words, the matrix multiplication breaks mathematically, and the code throws a fatal dimension mismatch error.To use traditional networks for text, engineers are forced to use padding or cropping: hacking every single sentence to be exactly, say, 50 words long by cutting off long essays or filling short sentences with meaningless zeros ([0, 0, 0, "Hello", "world"]).
    - How RNNs bypass this via unrolling: An RNN completely eliminates this restriction because it doesn't process the entire input sequence simultaneously in a single, massive matrix. Instead, it processes the sequence one index step at a time, using the exact same network cell recursively.Imagine an RNN cell as a physical conveyor belt loop inside a factory:Step 1: Word 1 passes through the cell. The cell updates its internal hidden state memory.Step 2: Word 2 passes through that same cell. The cell mixes Word 2 with the memory from Step 1.Step 3: Word 3 passes through that same cell...Because the network structure itself doesn't change, the conveyor belt doesn't care how many items pass along it. It simply keeps running the loop until the sequence runs out of data.

## Flaws of RNN
While mathematically elegant, standard RNNs suffer from a severe optimization bottleneck known as the vanishing gradient problem. 
  - During training, the backpropagation algorithm passes errors backward through time, multiplying weights repeatedly across every single time step. If the sequence is long and the weights are even slightly less than 1.0, the numbers shrink exponentially until they hit zero. Because the gradient vanishes, the early layers stop updating their weights. 
  - In practice, a standard RNN has an incredibly short-term memory and completely forgets what happened a mere 10 or 15 time steps ago.This specific limitation is what forced researchers to invent the LSTM (which uses mathematical gates to protect memory) and modern Transformers (which process all time steps simultaneously using attention matrices).
  - Explanation
     * The cause: An RNN uses the exact same weight matrix recursively at every single tick of the clock.
     * The math trap: To calculate updates for early time-steps, the backpropagation algorithm must multiply this shared weight matrix by itself for every step in the sequence (\(W^{T}\)).
     * The collapse: If those weight numbers are even slightly below 1.0 (or pass through crushing functions like Sigmoid), multiplying them 50 or 100 times over forces the gradient to drop exponentially down to absolute zero.
     * The outcome: The front of the network stops learning completely. The RNN suffers from severe short-term memory loss, remaining blind to any context established at the beginning of a long sequence.

## LSTM
A Long Short-Term Memory (LSTM) network is a specialized type of Recurrent Neural Network (RNN) designed explicitly to process sequential data where the order and timing of information matter, such as text sentences, spoken audio, or financial stock trends over time.Standard neural networks assume all inputs are independent, but LSTMs possess an internal memory that allows them to pass historical context forward along a timeline.
  * The problem it solves: Vanishing gradients Traditional RNNs suffer from a severe mathematical bottleneck called the vanishing gradient problem. As an RNN processes a very long sequence (like a long paragraph of text), it continuously multiplies numbers during backpropagation. If those numbers are small, the gradients shrink exponentially until they hit zero.As a result, the network completely forgets what happened at the beginning of the sequence. For example, if given the sentence "I grew up in France... [100 words later]... I speak fluent ______," a standard RNN cannot retain the context of "France" and fails to predict "French." An LSTM fixes this.
  * The core mechanism: The cell state highway The secret to the LSTM’s success is the Cell State (\(C_{t}\)), which acts like a continuous, unimpeded highway running straight down the entire length of the sequence. Information can flow along this highway with only minor linear interactions, allowing long-term memories to remain completely intact across hundreds of time steps. The network manages this memory using three specialized mathematical filters called gates.
  * The three internal gates Every LSTM cell uses a combination of sigmoid (σ) and tanh activation functions to regulate what enters, modifies, or leaves the cell state at each specific time step (t).
  
                    [ Cell State Highway (Ct-1 -> Ct) ]
                                  ▲
        ┌─────────────────────────┼────────────────────────┐
        │                         │                        │
 [ Input: xt ] ──► [ Forget Gate ] ──► [ Input Gate ] ──► [ Output Gate ] ──► [ Output: ht ]
        │                                                                          │
        └──────────────────────────────────────────────────────────────────────────┘

    - The forget gate (What to delete) This gate decides what historical information is no longer relevant and should be wiped from the memory.How it works: It looks at the new incoming data (\(x_{t}\)) and the previous hidden output (\(h_{t-1}\)). It passes them through a sigmoid function, outputting a number between 0 (completely forget this) and 1 (completely keep this).
    - The input gate (What to remember) This gate decides what new incoming information is worth adding to the cell state highway.How it works: It runs in two parts. A sigmoid layer decides which values to update, while a tanh layer creates a vector of new candidate values. These are multiplied together and added directly to the cell state highway to update the network's long-term memory.
    - The output gate (What to predict) This gate decides what parts of the internal memory should be surfaced as the immediate output for the current time step.How it works: It passes the updated cell state through a tanh function to scale the numbers between -1 and 1. It then multiplies this by a sigmoid filter (derived from the current input and previous hidden state) to output the final hidden state (\(h_{t}\)), which is sent to the next cell.
  * Real-world applications 
    - Natural language processing: Machine translation, sentiment analysis, and text generation.
    - Time-series forecasting: Predicting stock market trends, energy grid demands, or weather patterns.
    - Speech recognition: Converting spoken audio signals into text tokens sequentially.