## Types of Transformers

1. BERT - reads both directions for understanding tasks like Q&A, sentiment analysis
2. GPT - predicts next word for text generation, chatbots
3. BART - blends BERT and GPT for summarization, translation, text repair
4. Multi modal transformers - handles image, audio and video

## BERT Architecture

BERT (Bidirectional Encoder Representations from Transformers) is designed to understand the context of words in a sentence by looking at both the left and right sides simultaneously. It consists of three main stages: Input Embedding, stacked Transformer Encoder Layers, and Output/Head layers.
Here is an explanation of the layers in a BERT Base model (12 layers) with an example sentence: "Bank deposit".

1. The Input Embedding Layer: Before entering the encoder, text must be converted into numerical representations. This layer combines three types of embeddings to create a single 768-dimensional vector for each token:
  - Token Embeddings: Converts tokens (subwords) into vectors. Example: [CLS], Bank, deposit, [SEP]
  - Segment (Token Type) Embeddings: Distinguishes between sentence A and sentence B (used in Next Sentence Prediction).
   Example: If "Bank deposit" is one sentence, all tokens are marked as 'Segment 0'.
  - Position Embeddings: Tells BERT the order of tokens in the sentence (e.g., "Bank" is at pos 1, "deposit" at pos 2).

Output: A combined vector representation for every token that includes its meaning, position, and sentence pair context.
2. Transformer Encoder Layers (12 Blocks): BERT Base uses 12, and Large uses 24, identical transformer encoder blocks stacked on top of each other. Each block is responsible for capturing increasingly complex, contextual relationships.
Each encoder layer has two main sub-layers:

- Multi-Head Self-Attention Layer: This allows the model to look at other words in the input sequence to better understand a specific word. Example: In "I went to the bank to deposit money," the word "bank" needs context. The attention mechanism weighs "deposit" and "money" heavily, allowing "bank" to learn it means a financial institution, not a riverbank. BERT uses multiple heads (12 for base) to focus on different types of relationships simultaneously (e.g., one head focuses on verbs, another on nouns).
- Feed-Forward Network (FFN): The output of the self-attention layer is passed through a feed-forward neural network to further refine the representation of each token. It uses a GELU activation function to handle complex, non-linear relationships.
- Residual Connections & Layer Normalization: After both the self-attention and the FFN, there is a "residual connection" (adding the input to the output) followed by layer normalization. This prevents gradients from vanishing, making it possible to train deep models.

Output of Encoder Layers: 12 layers of contextualized embeddings where the word "bank" has been transformed from a general concept to a "financial bank" concept.

## The three-stage lifecycle of a BERT model

- Phase 1: Pre-training (Where MLM and NSP exist)This is the only time MLM and NSP are alive. Google or open-source researchers take a completely blank BERT architecture (initialized with random numbers) and stream billions of raw Wikipedia paragraphs through it.During this phase, special training heads (temporary linear layers) are attached to the very top of the 12-layer encoder stack. The goal is to to learn real grammar rules, synonyms, and context
  ```
    [ Raw Unlabeled Text ]
             │
             ▼
    [ 15% Masking Engine ] ──► Injects [MASK] tokens and pairs sentences (IsNext/NotNext)
             │
             ▼
    [ 12-Layer Encoder ]   ──► Computes bidirectional self-attention matrices
             │
             ▼
    [ PRE-TRAINING HEADS ] ──► MLM Head (predicts tokens) + NSP Head (predicts True/False)
             │
             ▼
    [ Loss Calculation ]   ──► Calculates errors to backpropagate and update encoder weights
  ```
- Phase 2: Fine-tuning (The transition step)Once the model becomes excellent at predicting masks and sentence logic, pre-training stops. The developer buys or downloads these trained weights.At this moment, the developer deletes the MLM and NSP training heads entirely. In their place, the developer welds a new, lightweight Task Head onto the exact same 12-layer encoder backbone. The developer trains the model again on a much smaller, specific dataset (like 5,000 product reviews). The encoder weights are slightly adjusted, and the new Sentiment Head learns how to read the [CLS] token to classify text.
  ```
            [ MLM Head ] + [ NSP Head ]  ◄── CHOPPED OFF & DELETED
                                   ▲
                                   │
  [ Small Labeled Dataset ] ──► [ 12-Layer Encoder ] ──► [ NEW Sentiment Task Head ]
                                  (Saved Weights)          (Initialized Randomly)
  ```
- Phase 3: Inference (The final production model)This is the model that is deployed to a server. The final compiled architecture file contains only the 12-layer encoder backbone and custom task layer.



## Masked Language Modeling (MLM)

MLM is the task that gives BERT its deep bidirectional powers. It acts like a fill-in-the-blank game. The process 

- Masking (15%): BERT takes a regular text sequence and randomly selects 15% of the tokens to be evaluated
- Sub-distribution: Of those chosen tokens, they aren't all treated the same way (to prevent the model from getting lazy during fine-tuning when no masks exist):80% are replaced with a literal [MASK] token.10% are replaced with a completely random word from the vocabulary.10% are left unchanged as a control mechanism.
- Prediction: The sentence passes through the Encoder layers. The model must use the surrounding words from both the left and right sides to predict what the hidden word was. Example
  ```
  Input sentence:  "The chef cooked a delicious meal for dinner."
  MLM Processing:  "The chef [MASK] a [RANDOM: brick] meal for dinner."
  Model Objective: Predict that position 3 is "cooked" and position 5 is "delicious". 
  ```



## Next Sentence Prediction (NSP)

While MLM teaches BERT how individual words relate to each other, NSP teaches BERT how entire sentences relate to one another, which is critical for tasks like Question-Answering and text summarization.The process

- Sentence Pairing: BERT wraps two sentences together using special structural tokens:It starts with a [CLS] (Classification) token.It separates the sentences with a [SEP] token.
- Dataset Split (50/50):IsNext (50%): Sentence B is the actual next sentence that immediately followed Sentence A in the original text.NotNext (50%): Sentence B is a completely random sentence pulled from a different document in the corpus
- Classification: The model processes the paired inputs. It evaluates the final vector hidden state output at the very first [CLS] token position and pushes it through a binary classifier to guess: Is this pair logically connected (IsNext) or random noise (NotNext)?
  ```
  [CLS] The sky is clear today . [SEP] The sun is shining brightly . [SEP] ──► Model predicts: IsNext (1.0)
  [CLS] The sky is clear today . [SEP] Pandas eat green bamboo stems . [SEP] ──► Model predicts: NotNext (0.0)
  ```



## BERT Limitations

1. Context Window Limit​: Maximum Token Limit of 512 (around 400 words) on default models – problematic to handle long documents efficiently​
2. Dimensionality Cost of Tokens​: Token dimensions in standard BERT model are of length 768 – storage and processing can be computationally expensive​
3. No Generative Capabilities​: Encoder only model which is purely based around semantic understanding of text​. Can generate a few words due to MLM implementation – but not for generating full sentences​. Integration with other models like GPTs are thus required​



## BERT Variants

1. RoBERTa(Robustly optimized BERT pretraining approach): Improves pre-training by changing the choice of tokens to be masked during iterations
2. ELECTRA(Efficiently Learned Encoder that Classifies Token Replacements): Selected token is replaced with wrong one(token corruption) instead of MASK
3. DistilBERT: Optimized for Inference
4. ALBERT: Designed for parameter efficiency
5. Span BERT: tailored for span-based tasks
6. CodeBERT: specialized for programming



## GPT Architecture

The GPT (Generative Pre-trained Transformer) model architecture is a "decoder-only" transformer, consisting of a stack of identical layers (often 12 to 96+ depending on the model size) that transform raw text into context-aware, probabilistic predictions. Each layer processes the information from the previous one, allowing the model to understand complex, hierarchical relationships in language. Here is a breakdown of the key layers and components in a GPT model, in order of data flow.

- Input Embedding Layer: This is the initial layer that converts raw text into numerical representations.
  - Function: Tokenizes input text, converts tokens into high-dimensional vectors (embeddings), and adds positional encodings to understand the order of words.
  - Example: For input "Cat sat," tokens might be [3452, 1201]. The embedding layer turns these into two, say, 768-dimensional vectors that represent the meaning of "cat" and "sat."
- Transformer Decoder Blocks (Stacked Layers): GPT is a "decoder-only" model, meaning it stacks many decoder blocks (e.g., 12 in GPT-2 small, 96 in GPT-3) to process context. Each block has two main sub-layers:
  - Masked Multi-Head Self-Attention Layer: This is the "secret sauce" of GPT, allowing it to understand context by relating each word to all other words in the sentence. Instead of one attention mechanism, it uses several ("heads") running in parallel, enabling it to focus on different aspects of language simultaneously (e.g., one head for grammar, one for coreference).
    - Function: Calculates "attention scores" to weigh the importance of other words in the sequence. It is "masked" because during training, the model cannot see future tokens (it only knows the words before it). This autoregression is possible due to causal language modeling (CLM) during pre-training allowing it to generate text left to right purely based on previous words
    - Example: In "The bank robbery," the attention mechanism helps "bank" relate strongly to "robbery" (financial institution) rather than a river bank, based on the surrounding context.
  - Position-wise Feed-Forward Network (FFN): After the attention mechanism gathers context, the data passes through a standard neural network.
    - Function: Processes the attention output to refine the representation of each token independently. It typically consists of two linear transformations with a GELU activation function in between.
    - Example: It takes the contextualized vector for "bank" and transforms it to better represent that it is specifically a "financial institution."
  - Residual Connections and Layer Normalization: "Add & Norm" steps occur around both the attention and feed-forward layers.
    - Residual Connection: Adds the input of a layer to its output. This helps prevent the "vanishing gradient" problem, allowing deeper models to train better.
    - Layer Normalization: Ensures the numerical values stay within a reasonable range, stabilizing the training.
- Final Linear and Softmax Layer: Once the data has passed through all stacked transformer blocks, the final layer produces the output.
  - Linear Layer: Maps the high-dimensional vector from the last transformer block back to a vector of the size of the model's vocabulary.
  - Softmax Layer: Converts the raw output scores (logits) from the linear layer into probabilities for every possible next token in the vocabulary.
    - Example: After processing "The cat sat on the...", the final layer outputs probabilities, perhaps: {"mat": 0.8, "rug": 0.1, "dog": 0.05, ...}. The model then selects the highest probability or samples from the top results.
- Summary Data Flow Example
  - Prompt: "What is"
  - Embedding:
     (Vector for "What"),
     (Vector for "is")
  - Transformer Blocks (xN):
    - Attention: "What" looks at "is" (and vice versa) to understand it's a question.
    - Feed-Forward: Refines the vector to represent "This is a question-initiating phrase."
  - Final Layer: Outputs high probability for "your" or "the" as the next word.

