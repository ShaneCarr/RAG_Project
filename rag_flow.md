```mathematica
   ┌─────────────────────┐
   │   Your Code Repo    │
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │ Chunker (150 lines) │
   └─────────┬───────────┘
             │  "def login_user..."
             ▼
   ┌─────────────────────┐
   │ Embedding Model     │  → converts text/code into vector
   └─────────┬───────────┘
             │  [0.023, -0.19, 0.77, ...]  (high-dim vector)
             ▼
   ┌─────────────────────────┐
   │ Vector Database (VDB)   │
   │ Stores: {file, chunk,   │
   │           vector, meta} │
   └─────────┬───────────────┘
             │
   User Query: "Where is Redis initialized?"
             │
             ▼
   ┌─────────────────────┐
   │ Query Embedding     │  → same embedding model
   └─────────┬───────────┘
             │
   ┌─────────────────────────────┐
   │ Nearest-Neighbor Search     │
   │ Finds top-k similar vectors │
   └─────────┬───────────────────┘
             │
             ▼
   ┌─────────────────────────────┐
   │ Retrieved Chunks            │
   │ e.g., redis_client.py       │
   └─────────┬───────────────────┘
             │
   ┌─────────────────────────────┐
   │ LLM Prompt Context           │
   │ "Use this code: ..."         │
   │ + User Question              │
   └─────────┬───────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │ LLM (Reasoning)     │
   └─────────┬───────────┘
             │
          Answer


```
