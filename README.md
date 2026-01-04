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
- `generation`: Parameters for text generation. This includes:
    - `temperature`, `top_k`, `top_p`: Standard sampling parameters.
    - `context_window_size`: The maximum number of tokens to keep in the conversation history to manage memory usage.
    - `speculative_steps`: The number of steps the model "looks ahead" to accelerate generation. A value > 0 enables speculative decoding.
- `hardware`: Set the computation device (`cpu`, `gpu`, `mps`).

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

1.  Place your text (`.txt`) and image (`.jpg`, `.png`) files for training in the `data` directory. The `DataLoader` will automatically pair files with the same basename (e.g., `story.txt` and `story.png`).
2.  (Optional) Adjust parameters in `config.json` to fit your needs.

### Step 3: Model Management

This project includes a system for managing, training, and running different models. All models are stored in their own subdirectories within the `models/` folder.

#### Training a New Model

To train a new model from scratch, you must give it a unique name. The script will create a new directory inside `models/` with this name and save the model weights, configuration, and training logs there.

```bash
python3 train.py --model-name <your-model-name>
```
*Example:* `python3 train.py --model-name my-first-model`

This will create `models/my-first-model/` and start the training process.

#### Resuming Training (Fine-tuning)

You can continue training a previously saved model. This is useful for fine-tuning or simply resuming an interrupted session. Use the `--resume-from` flag to specify which existing model to load, and the `--model-name` flag to define where to save the results of the new training session (you can use the same name to overwrite or a new name to create a fine-tuned version).

```bash
python3 train.py --resume-from <existing-model-name> --model-name <your-model-name>
```
*Example to continue training:*
`python3 train.py --resume-from my-first-model --model-name my-first-model`

*Example to fine-tune:*
`python3 train.py --resume-from my-first-model --model-name my-finetuned-model`

#### Generating Text

To generate text, you can either specify which model to use or have the script prompt you to choose from the available models.

**Option A: Specify the model directly**

```bash
python3 generate.py --model-name <your-model-name>
```
*Example:* `python3 generate.py --model-name my-first-model`

**Option B: Choose from a list**

If you run the script without specifying a model name, it will scan the `models/` directory and present a list of all available models for you to choose from.

```bash
python3 generate.py
```

The script will load the selected model and its associated configuration to run the agentic generation loop.

## Performance Optimization

This project includes several advanced features to optimize performance for both training and inference on local hardware.

### Hardware-Specific Scripts

The `scripts/` directory contains convenient wrapper scripts to run training and generation with hardware-optimized settings.

- **`train_cpu.sh`, `generate_cpu.sh`**: Runs on the CPU.
- **`train_gpu.sh`, `generate_gpu.sh`**: Runs on a discrete NVIDIA GPU.
- **`train_hybrid.sh`, `generate_hybrid.sh`**: Runs in a hybrid mode where the main model is on the GPU and the Long-Term Memory (LTM) is on the CPU.

These scripts automatically pass the correct `--hardware-strategy` to the underlying Python scripts.

*Example:*
```bash
# Train a model on the GPU
./scripts/train_gpu.sh --model-name my-gpu-model

# Generate text with the trained model
./scripts/generate_gpu.sh --model-name my-gpu-model
```

### Parameter-Efficient Fine-Tuning (LoRA & QLoRA)

LoRA (Low-Rank Adaptation) allows you to fine-tune the model by training only a small fraction of its parameters, significantly reducing memory requirements. QLoRA combines LoRA with 4-bit quantization for even greater memory savings.

**How to use:**

1.  **Enable LoRA in `config_train.json`**: Add or uncomment the `"lora"` section in your configuration file.
    ```json
    "lora": {
      "r": 8,
      "lora_alpha": 16,
      "target_modules": ["qkv_proj", "w1", "w2", "w3", "gate"],
      "lora_dropout": 0.05,
      "bias": "none"
    }
    ```
2.  **Enable QLoRA**: To use QLoRA, simply start the training script with the `--load-in-4bit` flag. This will load the base model in 4-bit and then apply the LoRA adapters.

*Example (Training with QLoRA on GPU):*
```bash
./scripts/train_gpu.sh --model-name my-qlora-model --load-in-4bit
```

### CPU Quantization Workflow

For users without a powerful GPU, this project provides a workflow to quantize a model for faster inference on the CPU.

**Step 1: Train your model normally.**

**Step 2: Run the optimization script.**
The `optimize_and_run_cpu.sh` script automates the process. It takes your trained model, creates a quantized version, and then immediately starts a generation session with the optimized model.

```bash
./scripts/optimize_and_run_cpu.sh --model-name <your-model-name>
```
*Example:* `./scripts/optimize_and_run_cpu.sh --model-name my-first-model`

This will create a new file `models/my-first-model/model_quantized_cpu.pt`. Subsequent generation runs can use this quantized model by adding the `--quantized` flag:

```bash
./scripts/generate_cpu.sh --model-name my-first-model --quantized
```
