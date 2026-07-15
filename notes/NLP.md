## NLP
Natural Language Processing (NLP) is the branch of artificial intelligence that bridges the gap between human communication and machine understanding, translating unstructured text into structured data.

## NLP Pipeline
1. Text preprocessing: Cleaning raw stringsBefore text can be fed into any neural network, it must undergo a standardized pipeline to remove noise and establish uniform formatting:
  - Tokenization: Breaking down a continuous string of text into individual units called tokens (words, sub-words, or characters). For example, "Coding is fun" becomes ["Coding", "is", "fun"].
  - Lowercasing: Converting all text to lowercase to ensure the network treats words like "The", "the", and "THE" as the exact same entity.
  - Stop word removal: Stripping out highly frequent, low-information filler words (such as "is", "am", "the", and "at") to allow the model to focus strictly on core keywords.
  - Stemming & Lemmatization: Reducing words down to their root form. Stemming uses crude rules to chop off word endings (e.g., "running" → "run"), while Lemmatization maps words back to their actual dictionary root using vocabulary analysis (e.g., "better" → "good").
2. Text vectorization: Turning words into math. Computers cannot read text; they can only perform matrix multiplication on numbers. Vectorization is the process of converting text tokens into numerical formats
  - Foundational Vectorization techniques (Traditional NLP)
    * One-Hot encoding: A basic matrix mapping where every unique word in the vocabulary gets its own dedicated row and column containing a single 1 and zeros everywhere else. This creates massive, inefficient, and sparse matrices that fail to capture any contextual relationships between words.
    * Bag of Words (BoW): Counts the absolute frequency of words within a document, completely ignoring grammar rules and word order.
    * TF-IDF (Term Frequency-Inverse Document Frequency): A statistical formula that weights a word's importance. It scores words higher if they appear frequently in a specific document, but penalizes them if they appear universally across your entire dataset, isolating unique keywords. Builds a rigid table where every column represents one unique word in your entire database vocabulary. If your dataset has 10,000 unique words, a 3-word sentence like "I love AI" becomes a vector of 10,000 numbers—where 3 slots have TF-IDF frequency scores, and the other 9,997 slots are filled with absolute 0 values. It remaines blind to syntax and synonyms.
  - Modern Embedding Techniques (RNN/LSTM Vectorization)
    * Word Embeddings (Word2Vec / GloVe(Global Vectors for Word Representation)/FastTrack): An advanced strategy that plots words into a dense, continuous high-dimensional vector space. Words with similar meanings or contexts end up positioned physically close to one another in this space, enabling mathematical relationships like Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen").
    * Tensorflow Embedding projector library can be used to visualize the word relationships. It uses TSNE for 2D/3D visualizations
3. Core downstream NLP tasks: Once text data is cleaned and vectorized, it is routed into specific architectures to solve practical tasks
  - Sentiment analysis: Categorizing the emotional tone of text into classes like positive, negative, or neutral (e.g., analyzing social media product reviews).
  - Named Entity Recognition (NER): Identifying and categorizing key proper nouns within text blocks, labeling items as specific categories like PERSON, ORGANIZATION, LOCATION, or DATE.
  - Sequence-to-Sequence (Seq2Seq): Transforming one variable-length sequence of words into an entirely different sequence, which powers applications like foreign language translation and automated document text summarization.

  ## NLP structural and architectural shift
  |Feature / Era|Traditional NLP|Early Deep Learning|Latest Transformers|
  |-------------|---------------|-------------------|-------------------|
  |Core Paradigm|Statistical frequency counting|Sequential tracking and static dense coordinates|Global parallel self-attention matrices|
  |Core Vectorization Methods|One-Hot Encoding, Bag of Words (BoW), TF-IDF|Word2Vec (CBOW/Skip-gram), GloVe, Static lookup tables|Native Token Embedding Layers with Positional Encoding|
  |Vector Structure|Sparse Vectors: High-dimensional, hollow arrays filled mostly with zeros.|Static Dense Vectors: Fixed, compact arrays of continuous decimals.|Dynamic Contextual Vectors: Dense arrays that morph based on surrounding tokens.|
  |Semantic Awareness|None: Blind to synonyms; relies on exact character matching.|Static: Captures general analogies, but assigns one permanent vector per word spelling.|Complete: Instantly differentiates homonyms (e.g., financial "bank" vs. river "bank").|
  |Context Tracking|Order Blind: Discards syntax and grammar entirely.|Sequential Tracking: Follows timelines word-by-word via RNN/LSTM memory cells.|Global Parallel Tracking: Ingests whole documents simultaneously using attention grids.|
  |Pros|Fast & light: Low CPU compute requirements.Transparent: Easy to audit and debug.Exact matching: Excellent for indexing technical terms.|Semantic clustering: Similar concepts group together.Dense memory: Compact arrays optimize hardware memory layout.|Dynamic meaning: Adapts perfectly to shifting contexts.Infinite memory: Solves temporal decay.GPU scaling: Maximises hardware throughput.|
  |Cons|Memory waste: Inefficient data structures.Context blindness: Cannot deduce underlying sentence meaning.|Static trap: Blind to contextual word shifts.Vanishing gradient: RNN cells lose memory over long paragraphs.|Massive resource costs: Demands substantial VRAM capacity.Quadratic compute complexity: Costs grow quadratically (\(O(N^2)\)) with text length.|