# Copyright 2025 Pathway Technology, Inc. / Graph BDH Extension
"""
Training & Demonstration Script for Graph BDH:
Trains Graph BDH with explicit synaptic state updates and spiking primitives on Shakespeare.
"""

import os
import time
from contextlib import nullcontext

import numpy as np
import requests
import torch
import torch.nn.functional as F

from graph_bdh import GraphBDH, GraphBDHConfig


# Configuration for Graph BDH prototype
CONFIG = GraphBDHConfig(
    n_layer=3,
    n_embd=32,                         # Low-rank / synaptic dimension d
    n_neurons=128,                     # High-dimensional neuron count n (n >> d)
    vocab_size=256,                    # Byte-level character vocabulary
    damping=0.05,                      # Synaptic trace decay rate u
    threshold_mode="spiking_binary",   # 'continuous', 'spiking_binary', 'spiking_rate'
    spiking_v_th=0.05,
    spiking_leak=0.85,
    surrogate="fast_sigmoid",
    surrogate_scale=4.0,
    synaptic_topology="hub_sparse",    # O(nd) sparse landmark-hub synaptic topology
    dropout=0.05,
)

BLOCK_SIZE = 64                        # Sequence length for BPTT unrolling
BATCH_SIZE = 16
MAX_ITERS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
LOG_FREQ = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading Tiny Shakespeare dataset...")
        res = requests.get(data_url)
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(res.text)


def get_batch(split="train"):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    split_idx = int(0.9 * len(data))
    data_split = data[:split_idx] if split == "train" else data[split_idx:]
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data_split[i : i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_split[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def main():
    fetch_data()
    print(f"Initializing Graph BDH on device: {device}")
    print(f"Configuration: n_neurons={CONFIG.n_neurons}, dim_synaptic={CONFIG.n_embd}, layers={CONFIG.n_layer}")
    print(f"Threshold mode: {CONFIG.threshold_mode}, Synaptic topology: {CONFIG.synaptic_topology}")

    model = GraphBDH(CONFIG).to(device)
    stats = model.get_sparsity_and_edge_stats()
    print(f"Total Physical Circuit Edges: {stats['total_physical_circuit_edges']}")
    print(f"Total Physical Synapses in G_s: {stats['total_physical_synapses']}")
    print(f"Average Wires/Synapses per Neuron: {stats['edges_per_neuron']:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    loss_acc = 0.0
    loss_steps = 0
    start_time = time.time()

    print("\nStarting training loop...")
    for step in range(MAX_ITERS):
        x, y = get_batch("train")
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stable surrogate backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_acc += loss.item()
        loss_steps += 1

        if (step + 1) % LOG_FREQ == 0:
            avg_loss = loss_acc / loss_steps
            elapsed = time.time() - start_time
            print(f"Step {step + 1:3d}/{MAX_ITERS:3d} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s")
            loss_acc = 0.0
            loss_steps = 0

    print("\nTraining completed! Generating text sample with persistent synaptic memory...")
    model.eval()
    prompt_str = "ROMEO:"
    prompt_ids = torch.tensor([[ord(c) for c in prompt_str]], dtype=torch.long, device=device)
    generated = model.generate(prompt_ids, max_new_tokens=60, temperature=0.8, top_k=5)

    output_bytes = bytes(generated[0].cpu().numpy().astype(np.uint8))
    output_text = output_bytes.decode("utf-8", errors="replace")
    print("\n--- Generated Sample ---")
    print(output_text)
    print("------------------------")


if __name__ == "__main__":
    main()
