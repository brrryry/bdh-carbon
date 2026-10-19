# Copyright 2025 Pathway Technology, Inc. / BabyLM TinyStories Training Pipeline
"""
BabyLM Training Pipeline for Graph BDH on `karpathy/tinystories-gpt4-clean`.

Features:
1. Loads `karpathy/tinystories-gpt4-clean` using Hugging Face datasets.
2. Fast tokenization using GPT-2 BPE tokenizer or byte-level fallback.
3. Colab-optimized training loop with:
   - Mixed precision (torch.amp.autocast) for CUDA (T4/V100/A100).
   - Gradient clipping and AdamW with cosine learning rate schedule.
   - Periodic checkpoint saving.
4. Integrated Neuromorphic Benchmarking:
   - Automatically runs NeuromorphicProfiler at evaluation time to report
     SynOps, activation sparsity, and energy efficiency multiplier.
"""

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from graph_bdh import GraphBDH, GraphBDHConfig
from neuromorphic_profiler import NeuromorphicProfiler, NeuromorphicHardwareSpecs


def parse_args():
    parser = argparse.ArgumentParser(description="Train Graph BDH BabyLM on TinyStories")
    # Architecture
    parser.add_argument("--n_neurons", type=int, default=512, help="High-dimensional neuron count n")
    parser.add_argument("--n_embd", type=int, default=64, help="Low-rank synaptic hub dimension d")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of layers L")
    parser.add_argument("--threshold_mode", type=str, default="spiking_binary", choices=["continuous", "spiking_binary", "spiking_rate"])
    parser.add_argument("--synaptic_topology", type=str, default="hub_sparse", choices=["hub_sparse", "power_law_sparse", "dense"])
    parser.add_argument("--damping", type=float, default=0.05, help="Synaptic decay rate u")
    parser.add_argument("--spiking_v_th", type=float, default=0.25, help="Membrane firing threshold V_th")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128, help="Context sequence length")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    return parser.parse_args()


class TinyStoriesStreamingDataset(IterableDataset):
    """
    Streams and tokenizes karpathy/tinystories-gpt4-clean on-the-fly.
    """
    def __init__(self, split: str = "train", block_size: int = 128, tokenizer=None):
        super().__init__()
        self.split = split
        self.block_size = block_size
        self.tokenizer = tokenizer

    def _get_stream(self):
        try:
            from datasets import load_dataset
            # Loads karpathy/tinystories-gpt4-clean in streaming mode
            ds = load_dataset("karpathy/tinystories-gpt4-clean", split=self.split, streaming=True)
            return ds
        except Exception as e:
            print(f"[Warning] Failed to load remote dataset ({e}). Using synthetic TinyStories stream for local execution.")
            return None

    def __iter__(self):
        stream = self._get_stream()
        buffer = []

        if stream is not None:
            for sample in stream:
                text = sample.get("story", "") or sample.get("text", "")
                if self.tokenizer:
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = list(text.encode("utf-8"))

                buffer.extend(tokens)
                while len(buffer) >= self.block_size + 1:
                    chunk = buffer[: self.block_size + 1]
                    buffer = buffer[self.block_size :]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
        else:
            # Fallback synthetic generator for local test runs
            dummy_text = "Once upon a time, there was a little dragon named Pip. Pip loved to fly high above the green forest. "
            tokens = list(dummy_text.encode("utf-8")) * 10
            while True:
                x = torch.tensor(tokens[: self.block_size], dtype=torch.long)
                y = torch.tensor(tokens[1 : self.block_size + 1], dtype=torch.long)
                yield x, y


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        print(f"Loaded GPT-2 tokenizer (vocab size: {vocab_size})")
        return tokenizer, vocab_size
    except Exception:
        print("Using byte-level fallback tokenizer (vocab size: 256)")
        return None, 256


def get_lr(step: int, max_steps: int, lr_max: float, lr_min: float = 1e-5, warmup_steps: int = 50):
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    tokenizer, vocab_size = get_tokenizer()

    config = GraphBDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_neurons=args.n_neurons,
        vocab_size=vocab_size,
        damping=args.damping,
        threshold_mode=args.threshold_mode,
        spiking_v_th=args.spiking_v_th,
        synaptic_topology=args.synaptic_topology,
    )

    model = GraphBDH(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    stats = model.get_sparsity_and_edge_stats()

    print("\n---------------- Model Configuration ----------------")
    print(f"Parameters: {param_count:,}")
    print(f"Neurons: {config.n_neurons} | Synaptic Hub Dim: {config.n_embd} | Layers: {config.n_layer}")
    print(f"Physical Circuit Edges: {stats['total_physical_circuit_edges']:,}")
    print(f"Physical Synapses in G_s: {stats['total_physical_synapses']:,}")
    print(f"Threshold Mode: {config.threshold_mode} | Synaptic Topology: {config.synaptic_topology}")
    print("------------------------------------------------------\n")

    train_dataset = TinyStoriesStreamingDataset(split="train", block_size=args.block_size, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Mixed precision context
    use_amp = "cuda" in device.type
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else torch.no_grad()

    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()

    print("Beginning BabyLM training loop...")
    for x, y in train_loader:
        if step >= args.max_steps:
            break

        x, y = x.to(device), y.to(device)

        # Learning rate schedule
        lr = get_lr(step, args.max_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        if use_amp:
            with amp_ctx:
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % args.eval_freq == 0:
            avg_loss = running_loss / args.eval_freq
            tok_per_sec = (args.batch_size * args.block_size * args.eval_freq) / (time.time() - start_time)
            print(f"Step {step:5d}/{args.max_steps:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")
            running_loss = 0.0
            start_time = time.time()

    # Save final model (saving config as a standard dict for PyTorch 2.6+ weights_only compatibility)
    import dataclasses
    ckpt_path = os.path.join(args.output_dir, "graph_bdh_babylm.pt")
    torch.save({"model_state": model.state_dict(), "config": dataclasses.asdict(config)}, ckpt_path)
    print(f"\nModel checkpoint saved to {ckpt_path}")

    # Neuromorphic Hardware Inference Benchmarking
    print("\nStarting Neuromorphic Hardware Simulation & Energy Profiling...")
    profiler = NeuromorphicProfiler(model, specs=NeuromorphicHardwareSpecs(name="Intel-Loihi-2-Class"))

    prompt_text = "Once upon a time"
    if tokenizer:
        prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    else:
        prompt_ids = torch.tensor([[ord(c) for c in prompt_text]], dtype=torch.long, device=device)

    generated, profiles = profiler.profile_generation(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=5)
    profiler.print_summary(profiles)

    # Decode sample
    if tokenizer:
        decoded_sample = tokenizer.decode(generated[0].tolist())
    else:
        decoded_sample = bytes(generated[0].cpu().numpy().astype(int)).decode("utf-8", errors="replace")

    print("--- Generated Sample ---")
    print(decoded_sample)
    print("------------------------")


if __name__ == "__main__":
    main()
