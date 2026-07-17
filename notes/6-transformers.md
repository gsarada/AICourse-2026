## Transformer Architecture

The Transformer architecture uses an Encoder-Decoder structure. The Encoder acts as the reader. It processes the raw source text and compresses it into a deep mathematical representation. The Decoder acts as the writer. It uses the encoder's compressed information to generate an output sequence (like a translation) token by token.

- High Level Architecture
[ Source Text ] ──► [ Tokenizer & Positional Encoding ]
           │
          ▼
 ┌──────────────────┐
 │  ENCODER STACK   │
 └──────────────────┘
           │
           │  (Cross-Attention Vector Pathway)
           ▼

 [ Target Text ] ──► [ Masked Input ] ──► [ DECODER STACK ] ──► [ Linear & Softmax Output ]

## Step 1 Tokenizing and Positional Encoding

Before data ever enters the Encoder stack, it must be transformed from raw, unreadable text strings into formatted numbers that carry both semantic meaning and positional structure.This preparation phase happens in three sequential operations inside the initial input pipeline:

[ Raw Source Text ] ──► [ 1. Tokenization ] ──► [ 2. Input Embedding Layer ] ──► [ 3. Positional Encoding ] ──► [ Into Encoder Layer 1 ]

1. **Tokenization (The slicing step)** - Computers cannot read characters directly, so a Tokenizer (using algorithms like Byte-Pair Encoding or WordPiece) slices the raw string into smaller sub-word units called tokens.
  - How it works: Common words stay whole, while rare words are broken into pieces (e.g., "transforming" might become ["transform", "##ing"]).
  - The output: Every token is mapped to a unique integer ID based on a pre-defined vocabulary lookup table.
  Example dimension: The sentence "Deep learning" becomes a 1D vector of IDs: [4102, 7381].
2. **Input embedding layer (The mapping step)** - The integer IDs pass into a native Embedding Matrix (a trainable lookup table initialized with random decimals at day zero).
  - How it works: The matrix matches each integer ID to a solid row of continuous decimals called a Static Dense Vector. This vector represents the word's base dictionary meaning before any sentence context is applied. Example dimension: If your model's embedding width (d_{\text{model}}) is 512, the 2-word sentence is transformed into a 2D tensor shape of (2, 512).
3. **Positional encoding (The sequencing step)** - Because the Transformer's self-attention mechanism processes all words simultaneously, it has no native concept of word order. To the Transformer, the phrases "The cat ate the mouse" and "The mouse ate the cat" look completely identical at this point.To fix this, the network generates a separate matrix of the exact same size (2, 512) containing unique mathematical wave coordinates using alternating Sine and Cosine functions:
  - Even index positions (2i): PE(pos, 2i) = sin( pos / 10000^(2i / d_model) )
    - Odd index positions (2i+1): PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
  - The final fusion: Instead of appending these coordinates as extra columns, the Positional Encoding matrix is element-wise added directly on top of the Input Embedding vectors.
  - The outcome: This mathematical operation seamlessly injects the spatial time-stamp of where each word sits in the sentence structure without expanding the tensor size.

The finalized matrix—now packing raw token values and word placement coordinates together—is passed straight into Encoder Layer 1 to begin self-attention processing.

## Step 2 The encoder component layers

The encoder takes the input tokens and maps their global relationships. Every single layer block inside the encoder stack contains two primary sub-layers:

1. Sub-layer 1: [Multi-head self-attention](#multi-head-attention) This layer calculates how every single word in the input sentence relates to every other word at the same time.It splits the input into three vector matrices: Queries (Q), Keys (K), and Values (V).It calculates a dot-product attention score: \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V This allows the network to automatically link pronouns to their correct nouns or verbs to their subjects across long distances.
2. Sub-layer 2: [Position-wise feed-forward network (FFN)](#feed-forward-network-ffn) After processing attention, the output passes into a standard dense fully-connected network. It processes each word position completely independently and identically using linear transformations and an activation function (like ReLU or GELU). This layer is where the network stores its learned global factual knowledge. [Supplemental plumbing](#the-add--norm-layer)
  - Residual connections: A skip-connection wraps around both sub-layers, allowing gradients to flow backward completely unhindered to prevent vanishing gradients.
  - Layer normalization (LayerNorm): Stabilizes and scales the mathematical outputs of the layers to speed up training.

## Step 3. The decoder component layers

The decoder generates output sequences. Because it generates text step-by-step, it must be blocked from "looking into the future" of the target phrase. To handle this constraint, every layer block inside the decoder stack contains three sub-layers:

1. Sub-layer 1: [Masked multi-head self-attention](#masked-multi-head-attention-details) This layer works exactly like the encoder's self-attention, but with a critical safety rule: Masking.It applies a triangular matrix that zeroes out all future tokens in the sequence.When generating word #3, the attention mechanism is physically blocked from seeing words #4, #5, or beyond, ensuring the model only uses past context to predict the next word.
2. Sub-layer 2: [Encoder-decoder cross-attention](#encoder-decoder-cross-attention-details) This is the bridge layer where the encoder and decoder communicate.It takes the Queries (Q) from the decoder's past text, but grabs the Keys (K) and Values (V) directly from the final output of the Encoder.This allows the decoder to look back at the original source sentence to figure out which source words are most relevant to the specific word it is trying to write next.
3. Sub-layer 3: [Position-wise feed-forward network (FFN)](#feed-forward-network-ffn) Identical to the encoder's FFN, this final sub-layer processes the blended contextual features before passing them up to the final output layer (a Linear layer and a Softmax function) to choose the single most likely token from the vocabulary.

## Understanding Query, Key, and Value in Attention Mechanism

When tokens pass into a Transformer layer, they are represented as a static input matrix X. 

1. To create Q, K, and V, the network multiplies this input X by three completely separate weight matrices (WQ, WK, WV).These weight matrices are filled with random numbers at day zero and are optimized via backpropagation during training:
  (Q=X * WQ | K=X * WK | V=X * WV).
   If a sentence has 5 tokens and the embedding size is 512, the resulting Q, K, and V arrays will all share the exact same matrix shape: (5, 512).
2. The mathematical pipeline of self-attention - Once these three matrices are generated, they are fed into the standard Scaled Dot-Product Attention formula: Attention(Q,K,V)=softmax(QK^{T}/sqrt{d_{k}})*V
  - operation 1: Inverting and multiplying QK^T (The matchup) The network multiplies the Query matrix by the transposed Key matrix. This performs a dot-product between every single word's Query and every single word's Key. What it measures: Geometric alignment. If a Query vector and a Key vector point in the exact same direction in high-dimensional space, their dot-product will yield a massive positive number. This indicates high contextual relevance.
  - operation 2: Scaling by 1/sqrt{d_{k}} (The stabilizer) The raw scores are divided by the square root of the dimension size of the keys d_{k}. Why it matters: In high dimensions (like 512 or 1024), dot-products can yield massive numbers. Pushing huge numbers into a softmax function flattens its gradient to zero, causing the vanishing gradient problem. Scaling down keeps the numbers small and training mathematically stable.
  - operation 3: Applying softmax (The probability map) The scaled scores pass through a softmax function row-by-row. This squashes all the numbers between 0.0 and 1.0, ensuring each horizontal row sums up to exactly 100%. This grid is the Attention Matrix, mapping exactly how much percentage focus word A should allocate to word B.
  - operation 4: Multiplying by V (The final blend) Finally, this attention percentage map is multiplied by the Value V matrix.Instead of keeping 100% of its own original meaning, each token takes a percentage of the values of all surrounding tokens. If a pronoun has a 85% attention link to a specific noun, 85% of that noun's Value vector is mathematically fused into the pronoun's output vector.

## Decoder Techniques for Text Generation

1. **Greedy Search** : selects the word with the highest probability at each step and feeds it back into the decoder to predict the next word.This continues until an end-of-sequence token is generated or a maximum length is reached.This makes it efficient and fast since it does not need to explore multiple options. However, Greedy Search has a key limitation—it can lead to suboptimal results. Since it only considers the most probable word at each step without looking ahead, it might miss better overall sequences. This can result in outputs that lack coherence or fluency in long texts.
2. **Top-k sampling**:  is a decoding strategy that refines text generation by limiting the model’s choices at each step. Instead of considering all possible words, the model focuses only on the top k most probable tokens, reducing the likelihood of selecting rare or nonsensical words. The value of k plays a crucial role in balancing diversity and coherence. A higher k allows more variety, introducing creativity but increasing the risk of incoherent sentences. In contrast, a lower k keeps the output predictable and fluent but may limit diversity, making the text repetitive. By carefully selecting k, we can control whether the model generates structured, logical responses or explores more diverse possibilities.
3. **Top-p sampling**: also known as nucleus sampling, is a more dynamic approach compared to top-k sampling. Instead of selecting a fixed number of top k tokens, this method considers the smallest subset of tokens whose cumulative probability exceeds a threshold p. This allows the model to adjust the number of candidate words based on their probabilities rather than an arbitrary limit. With top-p sampling, the model primarily selects high-probability words while still allowing lower-probability words to introduce diversity. This makes the generated text more flexible and natural compared to top-k sampling. By adjusting the threshold p, we can control how conservative or creative the text generation process is—lower p values lead to safer, more predictable outputs, while higher values introduce more variation and creativity. This approach helps maintain a balance between coherence and diversity, making it useful for applications that require a mix of structure and unpredictability.
4. **Beam search** is an optimization technique used to improve text generation by maintaining a fixed number of the top candidate sequences at each step. Instead of selecting just the highest probability word at each step, as in greedy search, beam search expands multiple promising candidates, or "beams," and evaluates them using a scoring function. This function considers both the sequence's likelihood and its potential to generate coherent output. By exploring multiple paths, beam search increases the chances of finding an optimal sequence, making it useful in applications like machine translation and text summarisation. However, it can still suffer from repetitiveness or miss out on highly creative alternatives if the beam width is too narrow.
5. **Sampling with Temperature**: When generating text, controlling randomness is crucial. This is where sampling with temperature comes in. The temperature parameter modifies the probability distribution of word selection, affecting how predictable or creative the generated text becomes. A higher temperature flattens the probability distribution, making the model more likely to pick less probable words. This increases diversity and creativity in text generation, leading to more varied outputs. On the other hand, a lower temperature sharpens the probability distribution, ensuring the model selects the most probable words, resulting in more deterministic and controlled text. This technique is especially useful when we want to adjust the balance between coherence and creativity in applications like storytelling, chatbot responses, or summarisation. By tweaking the temperature, we can guide the model to generate structured, predictable text or explore more novel and imaginative outputs.

## Evolution of Generative AI

1. Transformers for natural text processing
2. LLM's (BERT, GPT, T5) are built on top of transformer architectures.
3. Large Vision Models (LVMs) are AI systems designed to understand and generate visual content, such as images and videos, on a large scale. Like Large Language Models (LLMs), LVMs leverage deep learning architectures, such as convolutional neural networks (CNNs) or transformers, to process and analyse vast amounts of visual data.​​ By learning patterns and features from large datasets, LVMs can generate new, realistic, and coherent visual content. These models are used in various applications, from image generation and enhancement to video analysis and autonomous systems, providing powerful tools for tasks like computer vision, image synthesis, and object detection.​
4. Transformers are highly effective but come with significant computational costs, prompting the development of hybrid models that combine transformers with other architectures like CNNs (Convolutional Neural Networks) or RNNs (Recurrent Neural Networks) to improve efficiency.​
  - CNN + Transformer: This hybrid is widely used in computer vision, with models like Vision Transformers (ViTs) combined with ResNet architectures for image processing. The CNN component helps capture local spatial features, while the transformer handles long-range dependencies, allowing the model to learn richer visual representations
  - RNN + Transformer: This combination is particularly useful in tasks like speech recognition and time-series forecasting, where RNNs capture local sequential dependencies, and transformers model long-range dependencies. The hybrid approach benefits from the strengths of both architectures, balancing the processing of local and global information.​

## Sparse Transformers

Standard transformers, while highly effective, suffer from inefficiencies when dealing with long sequences due to their quadratic memory and computational requirements (compares every single token to every other token. If a text has N tokens, the model builds an N × N attention matrix. This means the computational cost and VRAM usage scale quadratically (O(N²))) from self-attention mechanisms. This limitation becomes especially problematic in tasks like document processing or long-range dependency modeling, where the input sequences can be extremely long. To address these challenges, Sparse Transformers introduce optimisations that reduce the memory footprint and improve computational efficiency, making them better suited for longer sequences. These optimisations allow the model to maintain performance while scaling more effectively to large input sizes.​ Examples:

- [Longformer](#longformer-local--global-attention-details): This model uses dilated attention, which selectively focuses on a subset of the input sequence, allowing it to process longer documents more efficiently. Longformer is particularly well-suited for document-level tasks such as question answering and summarisation
- [Linformer](#linformer-low-rank-matrix-projection-details): Linformer projects attention matrices to a lower-dimensional space, which significantly reduces the memory consumption and inference time. This approach makes it feasible to work with long sequences while maintaining the benefits of the transformer architecture
- [Reformer](#reformer-locality-sensitive-hashing-details): Reformer substitutes standard self-attention with locality-sensitive hashing (LSH), which helps improve efficiency by focusing attention on more relevant parts of the sequence, rather than processing all pairwise relationships. This results in both memory and computation savings, making it ideal for large-scale tasks.​

While these were brilliant research breakthroughs in 2020 to solve the quadratic scaling bottleneck O(N^2), they suffered from severe algorithmic trade-offs that caused them to lose the commercial evolutionary race.

## Transformers Challenges

1. **Hallucinations**: One of the major challenges in text generation is hallucination, where models generate outputs that sound plausible but are factually incorrect. This can occur when the model’s training data lacks sufficient context or accurate information, leading to the generation of fabricated or misleading details.​
2. **Biases**: Models often reflect the biases present in their training data, which may include racial, gender, or cultural biases. These biases can unintentionally influence the model’s responses, perpetuating harmful stereotypes or providing skewed information.
3. **Ethics**: The use of text generation models raises significant ethical concerns. Issues like misinformation, plagiarism, and copyright infringement can arise from the ability of models to generate content that closely mirrors existing works. Moreover, there are ethical considerations around the potential misuse of AI-generated content in manipulating opinions, creating fake news, or even automating harmful activities.​

## Multi-Head Attention

Multi-Head Attention is an improvement over standard self-attention, designed to capture complex, multi-faceted relationships between tokens in a sequence.

- Parallel Heads: Instead of one attention mechanism, it uses several "heads" in parallel, each with its own learned weight matrices.
- Different Perspectives: Each head focuses on different types of information, such as syntactic relationships or semantic context.
- Mechanism: It computes Scaled Dot-Product Attention (using queries, keys, and values) to determine which words are important to each other.
- Output: The outputs from all heads are concatenated and transformed, allowing the model to aggregate diverse information from the entire sequence.

## Feed-Forward Network (FFN)

The Feed-Forward Network is applied to each position (word) in the sequence independently and identically, following the attention mechanism.

- Structure: It consists of two linear transformations (dense layers) with a non-linear activation function (typically ReLU or GELU) in between.
- Transformation: The first layer projects the data into a higher-dimensional space, and the second projects it back to the original dimension, enhancing the model's capacity to represent complex features.
- Purpose: It acts as a processing unit that refines the contextual information gathered by the attention layer.

## The Add & Norm layer

In typical architectures, this block is used after both the Multi-Head Attention and the Feed-Forward Network sublayers to stabilize training, mitigate vanishing gradients, and enable deep network convergence by ensuring consistent data distributions and aiding gradient flow.

- Key Components and Purpose:
  - Add (Residual Connection/Skip Connection): The output of a sublayer f(x) is added to its original input x, forming f(x)+x. This allows gradients to propagate directly through the network during backpropagation, solving the vanishing gradient problem in deep models.
  - Norm (Layer Normalization): This step normalizes the sum from the "Add" operation, typically bringing inputs to a mean of zero and standard deviation of one across the feature dimension, reducing internal covariate shift.
- Why it matters:
  - Stability: It prevents values from becoming too small or too large, which can cause numerical instability (exploding gradients).
  - Efficiency: It makes training faster and more efficient.
  - Performance: It allows for training much deeper, more complex models without losing information.

## Encoder-Decoder cross attention (Details)
The Encoder-Decoder Cross-Attention layer uses the encoder's keys and values as a contextual map so the decoder knows which specific words from the user's original prompt it should translate or answer next.Without this cross-attention layer, the decoder would be completely blind to the user's input and would have no idea what it is supposed to be writing about.
    1. The structural mapping: Who brings what?Inside the cross-attention layer, the inputs are split up to create a communication bridge:
      - Queries (Q) come from the Decoder: The decoder brings the words it has already written so far in the output sequence. The Query asks: "I am currently writing the 4th word of my response. Which words in the original prompt are most relevant to what I need to write right now?"
      - Keys (K) come from the Encoder: The encoder brings the structural map of the user's original prompt. The Key acts like an index card saying: "I represent the word 'Apple' at position 3, and I am heavily linked to the concept of fruit.
      - "Values (V) come from the Encoder: The encoder brings the actual semantic content of the user's prompt. The Value holds the deep contextual meaning of the user's words.
    2. Walkthrough: Translating a sentence step-by-step To visualize how this generates the next new token, imagine a Transformer translating text from English (Encoder) to French (Decoder).
      - User Input (Encoder): "The black cat"
      - Target Output (Decoder): Wants to generate "Le chat noir"
      - The encoder reads "The black cat", creates Keys (K) and Values (V) for those words, and holds them still. Now, the decoder begins generation tick-by-tick:
      - Step 1: Generating the first word 
        * The decoder starts with a blank token: [Start].
        * The decoder projects a Query (Q): "What is the absolute beginning of this sentence pointing to in the English prompt?"
        * The cross-attention matrix multiplies this Query by the encoder's Keys ("The", "black", "cat").
        * The math finds the highest overlap with the English word "The".
        * The decoder pulls the Value (V) of "The", processes it through its feed-forward layer, and successfully outputs the first word: "Le".
      - Step 2: Generating the second word (Where the magic happens)
        * The decoder now has the sequence: [Start] "Le".
        * The decoder projects a new Query (Q) from its current position. Because French grammar places nouns before adjectives, the decoder needs to find the noun next.
        * The cross-attention matrix maps this Query against the encoder's Keys. Even though "black" comes next in English, the decoder's query mathematically aligns much stronger with the noun Key "cat".
        * The decoder extracts the Value (V) for "cat" and cleanly outputs: "chat".
      - Step 3: Generating the third word
        * The decoder now has: [Start] "Le" "chat".
        * It projects a Query looking for the descriptive modifier for the cat it just wrote.
        * The cross-attention dot-product matches heavily with the remaining unaddressed English Key: "black".
        * It extracts the Value for "black" and outputs the final token: "noir".

## Masked Multi-Head Attention (Details)

It is a critical component in Transformer decoders that allows them to process input sequences in parallel during training while ensuring that each token only attends to previous positions, not future ones. It prevents data leakage by applying a mask—typically a lower triangular matrix—to the attention scores before the softmax step, setting future tokens to -(infinity).

Key Aspects of Masked Multi-Head Attention

- Purpose: Ensures autoregressive causality, meaning the model generates text token-by-token, only basing its predictions on past, known tokens.
- Mechanism: During training, all words are fed at once. A masked attention mechanism prevents "cheating" by setting the attention scores for future positions to -(infinity). When normalized via softmax, these future tokens receive a probability of zero.
- Difference from Standard Self-Attention: Unlike standard Self-Attention (used in encoders), which allows a token to look at the entire sequence (past and future), masked attention strictly limits this view to the left context only.
- Parallelization: It maintains the parallel processing efficiency of transformers, allowing the entire sequence to be calculated at once, rather than sequentially.
- Applications: Primarily used in decoder-only models like GPT for language generation.
- How It Works
  - Calculate Scores: Queries (Q), Keys (K), and Values (V) are calculated as in standard attention.
  - Apply Mask: A lower-triangular matrix is applied to the score matrix QKT.
  - Softmax: The result is normalized to ensure that only past positions have non-zero attention weights.
  - Multi-Head Interaction: The process is repeated across multiple heads and then combined to allow the model to focus on different types of information.

## Longformer (Local + Global attention) (Details)
Released by Allen Institute for AI in 2020, Longformer drops quadratic attention in favor of a specialized attention pattern that scales linearly O(N), allowing it to process thousands of tokens efficiently.
- How it works: Instead of looking at everything, it combines two distinct attention strategies:
  - Sliding window (Local) attention: Each token only calculates attention scores for a small fixed window of neighboring tokens around it (e.g., 50 words to the left and 50 words to the right). This captures local context.
  - Global attention: The engineer flags specific critical tokens (like the [CLS] token or user question tokens). These selected tokens are allowed to look at every single other word in the entire document, and all words can look back at them.
- Best used for: Long-document tasks like document classification, coreference resolution, and long-form question answering.

## Linformer (Low-rank matrix projection) (Details)
Released by Meta (Facebook AI Research) in 2020, Linformer mathematically proves that the massive N × N attention matrix is highly redundant and can be heavily compressed into a low-rank matrix.
- How it works: Linformer uses linear projection matrices to physically downsample the length of the Key (K) and Value (V) matrices before performing self-attention.If a sequence length is N, Linformer projects K and V down to a tiny fixed dimension k (where k << N).The attention calculation shifts from an expensive N × N multiplication to a highly efficient N × k multiplication.
- The outcome: The computational complexity drops to strictly linear O(N) in both execution time and memory consumption.
- Limitations: Because the downsampling projection layers are hardcoded to a fixed maximum sequence length during training, a trained Linformer cannot dynamically adjust to handle variable text lengths during inference.

## Reformer (Locality-sensitive hashing) (Details)
Released by Google in 2020, Reformer introduces two distinct engineering tricks to maximize memory efficiency over massive sequences (up to 1 million tokens).
- Trick 1: Locality-Sensitive Hashing (LSH): In a standard attention matrix, most numbers wind up close to zero after passing through the softmax function, wasting GPU computation on irrelevant pairs. Reformer uses LSH clustering to group similar Query and Key vectors into shared statistical buckets. The model only runs attention calculations between words that land inside the same bucket, shrinking the workload to an efficient O(NlogN) scale.
- Trick 2: Reversible residual layers: Standard networks store the activations of every single layer in GPU memory during the forward pass so they can use them during backpropagation. Reformer implements reversible layers, meaning it recomputes the forward activations on the fly during the backward pass. This drops VRAM storage needs drastically, allowing massive sequence depths to fit onto standard graphics cards.