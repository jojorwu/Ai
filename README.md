# Transformer from Scratch using NumPy

This project is a decoder-only Transformer (GPT-style) implemented from scratch using only NumPy. It features an agent-based evolutionary training system and is capable of multimodal learning (text and images).

## Core Architectural Features

- **Modular Design**: Components like `Embedding`, `MultiHeadAttention`, `RMSNorm`, etc., are implemented as separate, reusable classes in the `nn_components/` directory.
- **Agent-based Evolutionary Learning**: Instead of a traditional single-model training loop, this project uses a population of agents. The `AgentManager` class controls their lifecycle: forking, specialization through experience, collaborative evaluation, and merging the best agents back into a base model.
- **Multimodality**: The model can process both text and images. A `VisionEncoder` converts images into patch embeddings, which are seamlessly integrated into the Transformer's input sequence.
- **Actor-Critic Architecture**: The Transformer has a dual-head design: a **Policy Head** for generating token logits and a **Value Head** for predicting a "usefulness" score of a sequence, crucial for the agent evaluation process.
- **Long-Term Memory (LTM)**: Each agent possesses a separate LTM module (a smaller MLP) that allows it to specialize. The LTM is updated during inference based on a "surprise metric," enabling continuous learning.
- **Bias-Free Architecture**: Following modern practices (e.g., Llama), linear layers in the Transformer blocks do not use bias vectors, enhancing training stability.
- **GPT-2 Style Weight Initialization**: Improves training stability by using a proven weight initialization scheme.
- **Grouped-Query Attention (GQA)**: Accelerates inference and reduces the KV cache size by allowing multiple query heads to share key/value heads.
- **RMSNorm**: Uses Root Mean Square Normalization for computational efficiency over standard LayerNormalization.
- **KV Caching**: Implemented for significantly faster text generation.
- **Rotary Positional Embeddings (RoPE)**: Improves the model's understanding of relative token positions.
- **SwiGLU Feed-Forward Network**: Uses the advanced Swish-Gated Linear Unit for better performance compared to standard FFNs.
- **Mixture of Experts (MoE)**: Can be configured to use MoE layers for a massive increase in parameters with only a small increase in computational cost during inference. Includes an auxiliary load-balancing loss.
- **Weight Tying**: The `Embedding` layer and the final `Linear` projection layer share weights, reducing the total parameter count.
- **AdamW Optimizer**: Uses the Adam optimizer with Decoupled Weight Decay for better regularization.
- **Advanced Training Techniques**:
    - **Gradient Accumulation**: Emulates a larger batch size on memory-constrained hardware.
    - **Gradient Clipping**: Prevents exploding gradients.
    - **Checkpointing**: Automatically saves and resumes training from checkpoints.
    - **Early Stopping**: Halts training if validation performance doesn't improve, saving the best model.
    - **Cosine Decay with Warmup LR Scheduler**: For more stable and effective training.
    - **Dropout**: For regularization during training.

## Configuration

All key parameters are managed in `config.json`.

- `model`: Core Transformer architecture settings (e.g., `d_model`, `num_layers`, `num_heads`).
- `vision`: Vision encoder parameters.
- `evolution`: Parameters for the agent-based training process (`num_agents`, `evolution_epochs`, `batch_size`, etc.).
- `optimizer`: AdamW optimizer settings.
- `ltm`: Long-Term Memory update parameters.
- `scheduler`: Learning rate scheduler settings.
- `generation`: Parameters for text generation (`temperature`, `top_k`, `top_p`, etc.).
- `hardware`: Set the computation device (`cpu`, `gpu`, `mps`).

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

1.  Place your text (`.txt`) and image (`.jpg`, `.png`) files for training in the `data` directory. The `DataLoader` will automatically pair files with the same basename (e.g., `story.txt` and `story.png`).
2.  (Optional) Adjust parameters in `config.json` to fit your needs.

### Step 3: Train the Model

To start the evolutionary training process, run the main training script:

```bash
python3 train.py
```

The script will read the configuration, initialize a population of agents, and run the evolutionary training loop. It saves the best-performing model's weights to the path specified by `best_model_path` in the config.

### Step 4: Generate Text

To generate text using the trained model, run:

```bash
python3 generate.py
```

This script loads the best model and generates a response based on the `start_text` and other parameters in the `generation` section of `config.json`. It also features an agentic loop that can use tools defined in `tools.py`.
