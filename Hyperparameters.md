### Table 3: System Hyperparameters and Configuration Settings

| Parameter                 | Value        |
|---------------------------|--------------|
| Retrieval Top-k           | 5            |
| Embedding Dimension       | 768          |
| Learning Rate             | 2e-5         |
| Batch Size                | 16           |
| Maximum Sequence Length   | 512          |
| Warmup Steps              | 1,000        |
| Training Epochs           | 3            |
| Dropout Rate              | 0.1          |
| Leader Cluster Size       | 50           |
| Keyword Corpus Size       | 10,000       |
| Similarity Threshold      | 0.75         |
| Graph Walk Length         | 3            |
| HNSW ef construction      | 200          |
| HNSW M                    | 16           |
| Temperature               | 0.07         |
| Negative Sampling Ratio   | 4:1          |
| Gradient Clipping         | 1.0          |
| Weight Decay              | 0.01         |
| Hardware                  | 4× Tesla V100|
| Framework                 | PyTorch 1.9.0|
| CUDA Version              | 11.2         |
