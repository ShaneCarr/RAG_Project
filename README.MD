```bash   
[Git Repo]
   |
   v
[Chunker] ---> [Embedding Model] ---> [Vector DB]
                                           |
                                    (nearest neighbors)
                                           |
[User Query] ---> [Embedding Model] ------>|
                                           v
                                        [LLM]
                                           |
                                        [Answer]


```
