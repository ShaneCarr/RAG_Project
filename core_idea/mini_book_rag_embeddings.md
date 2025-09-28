# 📘 Mini-Book: Neural Networks, Vectors, Transformers, and RAG

This is a self-contained background guide on how embeddings, vectors, transformers, and retrieval-augmented generation (RAG) work. It starts from the math foundations (linear algebra, cosine similarity), builds up through neural networks, and finishes with a practical example of using RAG on a codebase.

---

## 1. Foundations: Linear Algebra and Vectors

### What is a Vector?
A vector is just an ordered list of numbers. For example:

- 2D vector: `v = [3, 4]`
- 3D vector: `w = [1, -2, 0]`
- High-dimensional vector: `[0.12, -0.33, 0.98, …]`

Vectors represent many things:
- A position in space
- A direction and magnitude
- In machine learning: a compressed representation of meaning (semantic fingerprint).

### Vector Operations

1. **Addition**  
   `[1, 2] + [3, 4] = [4, 6]`

2. **Scalar Multiplication**  
   `2 × [1, 2] = [2, 4]`

3. **Dot Product (inner product)**  
   For vectors `a = [a1, a2, …, an]` and `b = [b1, b2, …, bn]`:
   ```
   a · b = a1*b1 + a2*b2 + ... + an*bn
   ```

   Example:  
   `[1, 2, 3] · [4, -1, 2] = 1*4 + 2*(-1) + 3*2 = 4 - 2 + 6 = 8`

4. **Norm (Length of a Vector)**  
   \\( ||v|| = sqrt(v · v) \\)

   Example:  
   `||[3, 4]|| = sqrt(3^2 + 4^2) = sqrt(25) = 5`

---

## 2. Cosine Similarity (Measuring Angle Between Vectors)

Cosine similarity measures how similar two vectors are by looking at the angle between them:

\\( \text{cosine}(u, v) = \frac{u · v}{||u|| * ||v||} \\)

- `1` → same direction (very similar)
- `0` → orthogonal (unrelated)
- `-1` → opposite

### Example

- `u = [1, 0]`, `v = [0, 1]`  
  - Dot product = 0 → Cosine = 0 → unrelated.

- `u = [1, 1]`, `v = [2, 2]`  
  - Dot product = 4, norms = sqrt(2), sqrt(8)  
  - Cosine = 4 / (1.41*2.82) ≈ 1 → almost identical.

---

## 3. Neural Networks Basics

A neural network is a function approximator: given input `x`, it outputs prediction `y`.  
It’s built from layers of neurons, where each neuron computes:

\\( y = f(Wx + b) \\)

- `W`: weights (learned parameters)
- `b`: bias
- `f`: activation (ReLU, sigmoid, etc.)

### Training with Backpropagation
- Forward pass: compute outputs.
- Loss function: measure error (e.g., cross-entropy, contrastive loss).
- Backprop: propagate error gradients backward.
- Update weights: gradient descent (tune `W` and `b`).

---

## 4. Word Embeddings: From Word2Vec to Transformers

### Word2Vec (Old School)
- Trains a lookup table: every word → vector (300 dims).  
- "Bank" has one vector, no matter context (river vs. money).

### Transformer Embeddings (Modern)
- Tokenize input (split words/code into tokens).
- Pass through transformer layers with **self-attention**.
- Self-attention formula:

\\( Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}}) V \\)

- Each token attends to every other token, so meaning changes with context.  
- Example: “bank” in *“fisherman by the bank”* ≠ “bank” in *“bank raised interest rates”*.

---

## 5. How Embedding Vectors Are Made

### Steps
1. **Tokenization** → split input (e.g., Python code).
2. **Initial Embedding Lookup** → tokens mapped to initial vectors.
3. **Positional Encoding** → add order information.
4. **Transformer Layers** → self-attention contextualizes tokens.
5. **Pooling** → combine into single vector (CLS token or mean pooling).
6. **Normalization** → scale vector to unit length.

### Training Objective
Contrastive learning: similar inputs → close vectors, dissimilar → far apart.

---

## 6. Retrieval and Vector Databases

### What’s Stored in a Vector Database?
Each entry = `{id, file, chunk, vector, metadata}`

Example:
| ID | File              | Chunk (snippet)                 | Vector (embedding)   |
|----|------------------|---------------------------------|----------------------|
| 1  | auth/login.py    | def login_user(username, pw): … | [0.021, -0.334, …]   |
| 2  | db/redis_client.py | class RedisPool: …             | [0.192, 0.077, …]    |

### Search Process
1. Query embedding = vector.  
2. Find nearest vectors (kNN search).  
3. Retrieve top-k chunks.  
4. Send to LLM along with query.

---

## 7. Worked Example: Cosine Similarity Search

Suppose we have:

- Query vector: `q = [1, 2]`
- Chunk A vector: `a = [2, 4]`
- Chunk B vector: `b = [-1, 0]`

Compute:

- Cosine(q, a) = (1*2 + 2*4) / (√5 * √20) = 10 / (2.236*4.472) = 1 → very similar  
- Cosine(q, b) = (1*-1 + 2*0) / (√5 * √1) = -1 / 2.236 ≈ -0.45 → dissimilar

So chunk A is chosen.

---

## 8. RAG (Retrieval-Augmented Generation)

Pipeline:
1. Break repo/docs into chunks.  
2. Embed chunks and store in vector DB.  
3. Embed query.  
4. Retrieve top-k relevant chunks.  
5. Feed into LLM with query.  
6. LLM produces grounded answer.

---

## 9. End-to-End Example

**Repo snippet:**
```python
def connect_to_redis(host, port):
    return Redis(host, port)
```

**Query:** "Where is Redis initialized?"

### Step 1: Embed Query
Query → vector `q = [0.12, -0.03, …]`

### Step 2: Retrieve
Nearest neighbor = chunk containing `connect_to_redis`.

### Step 3: Augmented Prompt
LLM gets:
```
Context:
File: db/redis_client.py
def connect_to_redis(host, port):
    return Redis(host, port)

Question: Where is Redis initialized?
```

### Step 4: Answer
LLM outputs:
```
Redis is initialized in db/redis_client.py by the connect_to_redis function,
which creates a Redis object from the given host and port.
```

---

# ✅ Summary
- **Vectors**: lists of numbers capturing meaning.  
- **Cosine similarity**: angle between vectors (closeness).  
- **Embeddings**: made by transformers using attention.  
- **Vector DB**: stores embeddings + metadata.  
- **RAG**: retrieval + LLM reasoning.  
- **Example**: search term → embedding → nearest code snippet → grounded answer.

---

This document should give you both the math grounding and the practical system-level understanding.


# Part I – Math Foundations

Before we dive deep into neural networks and embeddings, we need a strong foundation in the math they are built on.  
This section will review **linear algebra** (vectors, matrices) and **calculus** (derivatives, chain rule, gradients).  

---

## Chapter 1: Vectors and Linear Algebra

### 1.1 Scalars, Vectors, Matrices, Tensors
- **Scalar**: a single number (e.g., `5`).
- **Vector**: an ordered list of numbers (e.g., `[2, 3, 5]`).
- **Matrix**: a 2D array of numbers, like a table:
  ```
  [1 2 3]
  [4 5 6]
  ```
- **Tensor**: a generalization (3D, 4D, etc.), used heavily in deep learning.

### 1.2 Vector Operations
1. **Addition**  
   `[1, 2, 3] + [4, 5, 6] = [5, 7, 9]`

2. **Scalar Multiplication**  
   `2 × [3, -1, 4] = [6, -2, 8]`

3. **Dot Product**  
   \\( a · b = \\sum_i a_i b_i \\)  
   Example: `[1, 2, 3] · [4, -1, 2] = 4 - 2 + 6 = 8`

4. **Norm (Length of a Vector)**  
   \\( ||v|| = sqrt(v · v) \\)  
   Example: `||[3, 4]|| = sqrt(3^2 + 4^2) = 5`

5. **Cosine Similarity**  
   \\( cos(u, v) = (u · v) / (||u|| ||v||) \\)  
   Measures angle, not length.  

   Example:  
   - `u = [1, 0]`, `v = [0, 1]` → cos = 0 (orthogonal).  
   - `u = [1, 1]`, `v = [2, 2]` → cos = 1 (same direction).

---

## Chapter 2: Matrices

### 2.1 Matrix Basics
Matrix = 2D array.  
Example:
```
A = [1 2
     3 4]
B = [5 6
     7 8]
```

### 2.2 Matrix Addition
```
A + B = [1+5  2+6
         3+7  4+8]
       = [6 8
          10 12]
```

### 2.3 Matrix Multiplication
Important rule: `(m×n) * (n×p) = (m×p)`  

Example:
```
A = [1 2
     3 4]
B = [5 6
     7 8]

A * B = [1*5 + 2*7   1*6 + 2*8
         3*5 + 4*7   3*6 + 4*8]

      = [19 22
         43 50]
```

### 2.4 Identity and Inverse
- Identity matrix: acts like `1` in multiplication.  
  `I = [[1,0],[0,1]]`  
  `AI = A`  
- Inverse: `A * A^-1 = I` (not all matrices have one).  

---

## Chapter 3: Calculus for Machine Learning

### 3.1 Derivatives
Derivative = slope (rate of change).  
- If `f(x) = x^2`, then `f’(x) = 2x`.

Example: slope at x=3: `f’(3) = 6`.

### 3.2 Partial Derivatives
For multivariable functions.  
`f(x, y) = x^2 + y^2`  
- ∂f/∂x = 2x  
- ∂f/∂y = 2y  

At (x=1, y=2): gradient = (2, 4).

### 3.3 Chain Rule
If `y = f(g(x))`, then  
`dy/dx = f’(g(x)) * g’(x)`

Example:  
- f(u) = u^2, g(x) = 3x+1  
- y = (3x+1)^2  
- dy/dx = 2(3x+1)*3 = 6(3x+1)

### 3.4 Gradients and Backprop
- Gradient = vector of partial derivatives.  
- Used in backpropagation to update weights.  

**Example:**  
Suppose `Loss = (y_pred - y_true)^2`  
- y_pred = Wx + b  
- dLoss/dW = 2(y_pred - y_true) * x  
- dLoss/db = 2(y_pred - y_true)`  

This is how neural nets “learn”: by nudging W and b to reduce loss.

---

✅ With these basics, you can handle the math behind embeddings, neural nets, and transformers.


# Part II – Neural Networks

Neural networks are at the heart of modern embeddings, transformers, and LLMs.  
This section explains **what a neural net is, how it computes (forward pass), how it learns (loss + backpropagation), and a worked numeric example**.

---

## Chapter 4: The Neuron and Forward Propagation

### 4.1 The Neuron
A neuron takes input, applies weights, adds a bias, and passes it through an activation function.

Formula:  
\\( y = f(Wx + b) \\)

- `x`: input vector  
- `W`: weights (matrix)  
- `b`: bias (vector)  
- `f`: activation (ReLU, sigmoid, tanh, etc.)

### 4.2 Example: Simple Neuron
Suppose:
- Input: x = [2, 3]  
- Weights: W = [0.5, -0.2]  
- Bias: b = 0.1  
- Activation: ReLU(z) = max(0, z)

Step 1: Weighted sum  
z = 0.5*2 + (-0.2*3) + 0.1 = 1.0 - 0.6 + 0.1 = 0.5

Step 2: Activation  
y = ReLU(0.5) = 0.5

Output: 0.5

---

## Chapter 5: Layers and Networks

### 5.1 A Layer
A layer is just multiple neurons stacked.

Example:  
Input x = [1, 2]  
Weights W =  
```
[0.2  0.8]
[-0.5  1.0]
```
Bias b = [0.1, -0.2]

Forward pass:  
z = Wx + b =  
```
[0.2*1 + 0.8*2 + 0.1]
[-0.5*1 + 1.0*2 - 0.2]
```

= [1.9, 1.3]

Apply ReLU → [1.9, 1.3]

### 5.2 A Network
Stack layers to form a deep network.  
- Input layer: raw data (pixels, tokens).  
- Hidden layers: learned features.  
- Output layer: prediction.

---

## Chapter 6: Loss Functions and Training

### 6.1 Loss Functions
Loss measures how wrong the prediction is.

Examples:
- **Mean Squared Error (MSE):**  
  L = (y_pred - y_true)^2
- **Cross-Entropy (classification):**  
  L = -Σ y_true log(y_pred)

### 6.2 Gradient Descent
We adjust weights to minimize loss:
\\( W := W - η * ∂L/∂W \\)  
where `η` = learning rate.

---

## Chapter 7: Backpropagation

Backprop is how we compute gradients efficiently.

### 7.1 Chain Rule Refresher
If y = f(g(x)), then dy/dx = f’(g(x)) * g’(x).  
Backprop applies this repeatedly layer by layer.

### 7.2 Worked Example: Tiny Neural Net

Network:  
- Input x = 2  
- Weight W = 3  
- Bias b = 1  
- Activation f(z) = z (identity, no nonlinearity)  
- Prediction y_pred = Wx + b = 3*2 + 1 = 7  
- True y = 5  

Loss = (y_pred - y_true)^2 = (7 - 5)^2 = 4

**Step 1: Gradient wrt y_pred**  
∂L/∂y_pred = 2*(y_pred - y_true) = 2*(7-5) = 4

**Step 2: Gradient wrt W**  
y_pred = Wx + b  
∂y_pred/∂W = x = 2  
∂L/∂W = ∂L/∂y_pred * ∂y_pred/∂W = 4 * 2 = 8

**Step 3: Gradient wrt b**  
∂y_pred/∂b = 1  
∂L/∂b = 4 * 1 = 4

**Step 4: Weight Update**  
Suppose η = 0.1  
W_new = W - η*∂L/∂W = 3 - 0.1*8 = 2.2  
b_new = b - η*∂L/∂b = 1 - 0.1*4 = 0.6

### Result
- Before: W=3, b=1, prediction=7, loss=4  
- After one step: W=2.2, b=0.6, prediction=2.2*2+0.6=5.0, loss≈0

The network has “learned” in one step!

---

## Chapter 8: Deeper Networks and Backprop in Practice

In larger networks:
- Each layer computes activations.  
- Backprop uses chain rule layer by layer.  
- Libraries like PyTorch/TensorFlow handle this automatically.  

Key intuition: **gradients tell us which direction to tweak weights to reduce error**.

---

✅ With this, you understand forward pass, loss, and backpropagation with real numbers.  
Next: embeddings, transformers, and how modern models build vectors from text/code.


# Part III – Representations: From Word2Vec to Transformers

Now that we understand neural networks and backpropagation, we can explore **how text/code is turned into vectors**.  
This is the foundation for embeddings, transformers, and RAG.

---

## Chapter 9: Word Embeddings – Word2Vec

### 9.1 Motivation
- Computers don’t understand raw text.  
- We need a numeric representation of words.  
- Goal: words with similar meaning should have similar vectors.

### 9.2 Word2Vec Basics
Word2Vec (2013) learns embeddings by predicting context.

- **Skip-gram:** predict context words given a target word.  
- **CBOW (Continuous Bag of Words):** predict target word given its context.

Example (Skip-gram):  
Sentence: “The cat sat on the mat.”  
Target = “cat” → predict words like “the”, “sat”, “on”.

### 9.3 Word2Vec Training
- Each word starts as a random vector.  
- When two words appear together often, their vectors are nudged closer.  
- Result: semantically similar words are close in vector space.

### 9.4 Limitations
- Each word has only ONE vector.  
- Context ignored (bank = river bank vs. money bank).

---

## Chapter 10: Transformers – Contextual Embeddings

Modern embeddings fix Word2Vec’s problem using **transformers**.

### 10.1 Tokenization
Input text is split into tokens (subwords, not always whole words).  
Example: “connecting” → [“connect”, “##ing”]

### 10.2 Positional Encoding
Transformers don’t know order naturally.  
So we add position vectors (sinusoidal or learned).

Example (dim=4):  
```
Token: "cat" → [0.12, -0.3, 0.7, 0.1]
Position 5 → [0.8, 0.5, -0.2, 0.3]
Combined → [0.92, 0.2, 0.5, 0.4]
```

### 10.3 Self-Attention
Core idea: each token looks at others to decide what’s important.

Formula:  
\\( Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}}) V \\)

- Q (query), K (key), V (value) are projections of token vectors.  
- Each token computes similarity with others (QK^T).  
- Softmax turns it into weights.  
- Weighted sum of V gives new representation.

### 10.4 Example of Attention
Sentence: “The bank raised interest rates.”  
- “bank” attends strongly to “interest”, “rates”.  
- Representation shifts toward financial meaning.

Sentence: “The fisherman sat by the bank.”  
- “bank” attends to “fisherman”, “sat”.  
- Representation shifts to river meaning.

So embeddings are **contextual**.

### 10.5 Multi-Head Attention
We don’t use just one attention, but many in parallel (“heads”).  
Each head learns different relations: syntax, semantics, long-range deps.

---

## Chapter 11: Transformer Embedding Output

### 11.1 Token Representations
After many layers, each token has a context-aware vector.

Example:  
- “cat” in “The cat sat” → vector A  
- “cat” in “Black cat magic” → vector B  
Vectors differ because context differs.

### 11.2 Sequence Embedding (Pooling)
For embeddings (e.g., OpenAI’s `text-embedding-3-large`):  
- We squash all token vectors into one vector.  
- Methods: [CLS] token, mean pooling.

Output: a **single high-dim vector (768, 1536, 3072 dims)** representing the whole text.

---

## Chapter 12: Training Modern Embeddings

### 12.1 Contrastive Learning
Instead of predicting words, modern embeddings are trained to bring similar pairs closer.

- Positive pair: (“def connect_to_redis”, “How to connect to Redis in Python”)  
- Negative pair: (“def connect_to_redis”, “Sorting an array in C++”)

Loss function (InfoNCE / contrastive loss):  
\\( L = -\log \frac{\exp(sim(q, p))}{\sum_{n} \exp(sim(q, n))} \\)

Where sim = cosine similarity.

### 12.2 Result
Embedding space clusters by meaning.  
- All “Redis” functions cluster.  
- All “auth” code clusters.  
- “user_id” references cluster.

---

✅ With transformers, embeddings are **contextual, semantic, and high-dimensional**.  
This is what we store in vector databases and use for retrieval.


# Part IV – Retrieval and Vector Databases

Now that we know how embeddings are created, let’s see how they are **used**.  
The core idea: embeddings turn search into a geometry problem.

---

## Chapter 13: Cosine Similarity and Distance Metrics

### 13.1 Cosine Similarity
Formula:  
\\( cos(u, v) = \frac{u · v}{||u|| \, ||v||} \\)

Interpretation:
- Measures angle between vectors.  
- Ignores magnitude (length).  
- Perfectly aligned → 1, orthogonal → 0, opposite → -1.

**Example:**  
u = [1, 2], v = [2, 4]  
- Dot = 1*2 + 2*4 = 10  
- Norms = √5, √20  
- cos = 10 / (2.236*4.472) = 1 → identical direction.

### 13.2 Euclidean Distance
Formula:  
\\( d(u, v) = \sqrt{\sum_i (u_i - v_i)^2} \\)

Example:  
u = [1, 2], v = [2, 4]  
d = √((1-2)^2 + (2-4)^2) = √(1+4) = √5 ≈ 2.24

### 13.3 When to Use Which?
- Cosine similarity: good for text/code (direction matters more than length).  
- Euclidean: good when magnitude encodes info.  
- In practice, cosine similarity is standard for embeddings.

---

## Chapter 14: Nearest Neighbor Search

### 14.1 Brute Force
- Compare query vector to all stored vectors.  
- Compute cosine similarity.  
- Pick top-k.  
- Fine for small DBs (like 30k LOC codebase).

### 14.2 Example: kNN Search
Query q = [1, 2]  
Database:  
- A = [2, 4] → cos=1.0  
- B = [-1, 0] → cos=-0.45  
- C = [0, 1] → cos≈0.89  

Top-2 neighbors = A, C.

### 14.3 Approximate Nearest Neighbor (ANN)
For millions of vectors, brute force is too slow.  
ANN algorithms speed it up:  
- **HNSW (Hierarchical Navigable Small Worlds)**: builds a graph of vectors.  
- **IVF (Inverted File Index)**: clusters then searches only relevant clusters.  
- **PQ (Product Quantization)**: compresses vectors for fast comparison.

Libraries: **FAISS, Annoy, ScaNN**.  
Databases: **Pinecone, Weaviate, Milvus, pgvector**.

---

## Chapter 15: Vector Databases

### 15.1 What’s Stored?
Each entry has:  
- **id** (unique identifier)  
- **vector** (embedding, e.g., 1536 floats)  
- **metadata** (file name, chunk text, tags)  

Example row:
```
{
  "id": 42,
  "file": "db/redis_client.py",
  "chunk": "def connect_to_redis(host, port): ...",
  "vector": [0.192, 0.077, ..., -0.051],
  "metadata": {"language": "python", "commit": "a1b2c3"}
}
```

### 15.2 Query Process
1. User query → embedding vector.  
2. ANN search → top-k nearest vectors.  
3. Retrieve chunks.  
4. Send to LLM.  

### 15.3 Determinism
- Embedding model is deterministic: same input → same vector.  
- Vector DB search is deterministic in brute force, approximate in ANN (but very close).

---

✅ At this point: we can embed text/code, store vectors, and retrieve similar chunks.  
Next: put it all together with Retrieval-Augmented Generation (RAG).


# Part V – Retrieval-Augmented Generation (RAG)

We’ve learned how embeddings are made, how they’re stored in vector databases, and how retrieval works.  
Now let’s put it all together into the **RAG pipeline**.

---

## Chapter 16: The RAG Pipeline

RAG = **Retrieval-Augmented Generation**.  
It enhances an LLM by providing it with external knowledge retrieved at query time.

### 16.1 Steps
1. **Ingest Data**
   - Break repo/docs into chunks (e.g., 150 lines of code).  
   - Embed each chunk.  
   - Store in vector DB with metadata.

2. **Ask a Question**
   - User query → embed into vector.

3. **Retrieve**
   - Use vector DB (cosine similarity / ANN) to find top-k similar chunks.

4. **Augment**
   - Inject retrieved chunks into the LLM’s prompt.

5. **Generate**
   - LLM uses both pretrained knowledge + retrieved context.  
   - Output is grounded in your data.

---

## Chapter 17: Worked Example (Codebase)

### 17.1 Code Snippet
File: `db/redis_client.py`
```python
def connect_to_redis(host, port):
    return Redis(host, port)
```

### 17.2 Query
"Where is Redis initialized?"

### 17.3 Embedding the Query
- Query embedding = vector q = [0.12, -0.03, …]

### 17.4 Retrieval
- Vector DB finds nearest neighbor = chunk with `connect_to_redis`.  
- Cosine similarity ≈ 0.98 → strong match.

### 17.5 Augmented Prompt to LLM
```
Context:
File: db/redis_client.py
def connect_to_redis(host, port):
    return Redis(host, port)

Question: Where is Redis initialized?
```

### 17.6 LLM Answer
```
Redis is initialized in db/redis_client.py by the connect_to_redis
function, which creates a Redis object from the given host and port.
```

---

## Chapter 18: Algorithms Behind RAG

### 18.1 Chunking
Algorithm:  
```
for file in repo:
  for i in range(0, len(file), chunk_size):
    chunk = file[i : i+chunk_size]
    embed(chunk)
    store(chunk, vector, metadata)
```

### 18.2 Retrieval (Cosine Similarity)
```
similarity(u, v) = (u · v) / (||u|| ||v||)
```

### 18.3 ANN Search (HNSW)
- Build graph of vectors.  
- Search by navigating graph from entry point to nearest neighbors.  
- Much faster than brute force.

### 18.4 Prompt Construction
```
prompt = "Answer the question using only this context:

"
for chunk in retrieved_chunks:
    prompt += chunk + "

"
prompt += "Question: " + user_query
```

---

## Chapter 19: Determinism and Storage

- Embedding generation is **deterministic** (same model + same text → same vector).  
- Retrieval is deterministic if brute force, approximate if ANN (but >99% accurate).  
- Vector DB stores:  
  - Chunk text  
  - Vector (list of floats)  
  - Metadata (file path, language, commit hash, etc.)  

---

## Chapter 20: Benefits of RAG

- **Accuracy**: reduces hallucination.  
- **Freshness**: no need to retrain model for new data.  
- **Scalability**: can handle millions of chunks.  
- **Control**: only feed trusted data to LLM.  
- **Traceability**: can cite sources (retrieved chunks).

---

## Chapter 21: Future Directions

- **Real-time updates**: auto-reindexing repos/docs on commit.  
- **Hybrid search**: combine keyword + semantic search.  
- **Multimodal RAG**: not just text/code, but also images, audio, tables.  
- **Interactive**: LLM asks *follow-up retrieval queries* to refine answers.

---

# ✅ End-to-End Recap

- You have a 30k LOC repo.  
- You chunk it, embed it, store vectors.  
- A query is embedded → nearest neighbors retrieved.  
- The LLM sees both query + chunks.  
- Answer is precise, grounded, and explainable.

This is the essence of **RAG**:  
**External memory (vector DB) + reasoning engine (LLM).**

---

# Appendix: Mini Example with Numbers

Suppose:

- Query embedding q = [1, 2]  
- DB has:
  - Chunk A = [2, 4]  
  - Chunk B = [0, 1]  
  - Chunk C = [-1, 0]

Compute cosine:

- cos(q, A) = 1 → perfect match  
- cos(q, B) = (1*0 + 2*1) / (√5*√1) = 2 / 2.236 = 0.89  
- cos(q, C) = -1 / 2.236 = -0.45

So we retrieve A, B.  
Augment LLM prompt with A, B.  
Answer is based on actual code.

---

✅ You now have an **end-to-end understanding** of how embeddings, transformers, vector DBs, and RAG work together.



# Appendix – Deep Dives

This appendix provides deeper dives into calculus, backpropagation, matrix calculus, and implementation examples.

---

## A1. Calculus Refresher: Derivatives and Chain Rule

### A1.1 Simple Derivative
f(x) = x^2  
f’(x) = 2x

At x=3: f’(3) = 6

### A1.2 Chain Rule
If y = f(g(x)), then  
dy/dx = f’(g(x)) * g’(x)

Example:  
f(u) = u^2, g(x) = 3x+1  
y = (3x+1)^2  
dy/dx = 2(3x+1)*3 = 6(3x+1)

---

## A2. Backpropagation Worked Example

### A2.1 Network Setup
- Input: x=1  
- Weight: W=2  
- Bias: b=1  
- Activation: ReLU  
- True label: y_true=4

Forward pass:  
z = Wx+b = 2*1+1=3  
y_pred=ReLU(z)=3  
Loss=(y_pred-y_true)^2=(3-4)^2=1

### A2.2 Backward Pass
∂Loss/∂y_pred = 2*(3-4) = -2  
∂y_pred/∂z = 1 (since ReLU’(3)=1)  
∂Loss/∂z = -2*1 = -2  
∂z/∂W = x=1  
∂z/∂b = 1

So:  
∂Loss/∂W = -2*1 = -2  
∂Loss/∂b = -2*1 = -2

### A2.3 Update
Learning rate η=0.1  
W_new=2-0.1*(-2)=2.2  
b_new=1-0.1*(-2)=1.2

Prediction improves.

---

## A3. Matrix Calculus in Neural Nets

For vectorized operations:

y = Wx + b  
Loss = (y - y_true)^2

Gradients:  
∂Loss/∂W = 2(y-y_true)x^T  
∂Loss/∂b = 2(y-y_true)

This is why frameworks use matrix calculus—it’s concise.

---

## A4. Numerical Backprop Example with Numpy

```python
import numpy as np

# Inputs and weights
x = np.array([1.0, 2.0])
W = np.array([0.5, -0.3])
b = 0.1
y_true = 1.0
lr = 0.1

# Forward pass
z = np.dot(W, x) + b
y_pred = z  # linear
loss = (y_pred - y_true)**2

# Backward pass
dL_dy = 2*(y_pred - y_true)
dL_dW = dL_dy * x
dL_db = dL_dy

# Update
W -= lr * dL_dW
b -= lr * dL_db

print("Updated W:", W)
print("Updated b:", b)
```

---

## A5. PyTorch Example with Autograd

```python
import torch

# Setup
x = torch.tensor([1.0, 2.0])
W = torch.tensor([0.5, -0.3], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
y_true = torch.tensor(1.0)
lr = 0.1

# Forward pass
y_pred = torch.dot(W, x) + b
loss = (y_pred - y_true)**2

# Backward pass
loss.backward()

# Gradient descent update
with torch.no_grad():
    W -= lr * W.grad
    b -= lr * b.grad

print("Updated W:", W)
print("Updated b:", b)
```

---

## A6. Determinism of Embeddings

- Same text/code + same embedding model → same vector (deterministic).  
- Floating point differences possible, but negligible.  
- Retrieval is deterministic in brute force; ANN is approximate (but >99% accurate).

---

✅ This appendix gives you deeper math and hands-on examples.  
With this, you should be able to both **understand the theory** and **try simple implementations**.



# Part VI – Practical Guide: Build a RAG System

Now that we’ve covered theory, let’s build a **mini RAG pipeline**.  
We’ll use Python, OpenAI embeddings, and a simple vector database (FAISS).

---

## Step 1: Install Requirements

```bash
pip install openai faiss-cpu
```

---

## Step 2: Chunk the Data

Suppose you have a repo with code files. We’ll break them into chunks.

```python
import os
from pathlib import Path

def load_repo(path):
    files = []
    for p in Path(path).rglob("*.py"):
        with open(p, "r", encoding="utf-8") as f:
            files.append((str(p), f.read()))
    return files

def chunk_code(filename, text, max_lines=100):
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        yield {
            "file": filename,
            "content": "\n".join(lines[i:i+max_lines])
        }
```

---

## Step 3: Generate Embeddings

```python
import openai

def embed_text(text):
    resp = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp["data"][0]["embedding"]
```

---

## Step 4: Store in FAISS (Vector DB)

```python
import faiss
import numpy as np

# Create FAISS index
dimension = 3072  # depends on model
index = faiss.IndexFlatL2(dimension)

chunks = []
vectors = []

for file, text in load_repo("my_repo"):
    for chunk in chunk_code(file, text):
        vec = embed_text(chunk["content"])
        chunks.append(chunk)
        vectors.append(vec)

vectors_np = np.array(vectors).astype("float32")
index.add(vectors_np)
```

---

## Step 5: Query the System

```python
def search(query, top_k=3):
    q_vec = np.array(embed_text(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0]]

results = search("Where is Redis initialized?", top_k=2)
for r in results:
    print(r["file"], "->", r["content"][:200])
```

---

## Step 6: Augment LLM Prompt

```python
def answer_question(query):
    context_chunks = search(query, top_k=3)
    context_text = "\n\n".join([f"{c['file']}\n{c['content']}" for c in context_chunks])

    prompt = f"""
    You are a code assistant. Use the following context to answer the question.

    Context:
    {context_text}

    Question: {query}
    """

    resp = openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["choices"][0]["message"]["content"]

print(answer_question("Where is Redis initialized?"))
```

---

## End-to-End Flow Recap

1. **Chunk** code/docs into small pieces.  
2. **Embed** each chunk with an embedding model.  
3. **Store** in vector database (FAISS, Pinecone, Weaviate).  
4. **Query**: embed the question and search for nearest chunks.  
5. **Augment**: add retrieved chunks to the LLM prompt.  
6. **Generate**: LLM answers grounded in your data.

---

## Example Output

Query: *“Where is Redis initialized?”*  
Retrieved chunk:
```python
def connect_to_redis(host, port):
    return Redis(host, port)
```
LLM Answer:  
```
Redis is initialized in db/redis_client.py inside the function connect_to_redis,
which constructs a Redis object using host and port.
```

---

# ✅ Final Notes

- With ~30k LOC, this system is very feasible.  
- FAISS can handle millions of vectors, so scale is not an issue.  
- You can extend this with:  
  - Real-time updates (watch repo for changes).  
  - Metadata filters (search only in `auth/` folder).  
  - Hybrid search (keywords + embeddings).

This practical guide shows you how to **actually build a working RAG system** end-to-end.

