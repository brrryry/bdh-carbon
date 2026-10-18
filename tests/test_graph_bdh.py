# Copyright 2025 Pathway Technology, Inc. / Graph BDH Test Suite
"""
Unit and Integration Tests for Graph BDH:
1. Graph Circuit Factorization (H_e, H_i) & 2-step signal propagation (G = H^2).
2. Explicit Synaptic Memory Matrix (sigma_{t,l}) & Local Hebbian updates.
3. Spiking / Thresholding Primitives with Surrogate Gradients.
4. Full Model End-to-End Sequence Forward & Backward (BPTT).
5. State-preserving Autoregressive Generation.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from graph_bdh import (
    FastSigmoidSurrogate,
    ATanSurrogate,
    SpikingIntegrateAndFire,
    SparseLinearCircuit,
    SynapticMemory,
    GraphBDHConfig,
    GraphBDH,
)


def test_fast_sigmoid_surrogate_gradient():
    """Validates that FastSigmoidSurrogate computes step function forward and smooth surrogate gradient backward."""
    v = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0], requires_grad=True)
    v_th = 0.0
    scale = 4.0

    # Forward: Heaviside step
    s = FastSigmoidSurrogate.apply(v, v_th, scale)
    expected_forward = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.allclose(s, expected_forward), f"Forward mismatch: {s} vs {expected_forward}"

    # Backward: grad = scale / (1 + |scale * (v - v_th)|)^2
    loss = s.sum()
    loss.backward()
    expected_grad = scale / ((1.0 + (scale * (v - v_th)).abs()) ** 2)
    assert torch.allclose(v.grad, expected_grad, atol=1e-5), f"Grad mismatch: {v.grad} vs {expected_grad}"


def test_atan_surrogate_gradient():
    """Validates that ATanSurrogate computes step function forward and atan derivative backward."""
    v = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
    v_th = 0.0
    scale = 2.0

    s = ATanSurrogate.apply(v, v_th, scale)
    expected_forward = torch.tensor([0.0, 1.0, 1.0])
    assert torch.allclose(s, expected_forward)

    loss = s.sum()
    loss.backward()
    expected_grad = (scale / math.pi) / (1.0 + (scale * (v - v_th)) ** 2)
    assert torch.allclose(v.grad, expected_grad, atol=1e-5)


def test_spiking_threshold_modes():
    """Tests continuous rectified thresholding, event-driven binary pulses, and rate-coded spiking."""
    n = 16
    x_prev = torch.ones(2, n)
    exc = torch.tensor([[0.5] * n, [1.5] * n])
    inh = torch.tensor([[0.2] * n, [0.3] * n])

    # 1. Continuous mode: (X_e - X_i)^+
    snn_cont = SpikingIntegrateAndFire(mode="continuous")
    x_next, _ = snn_cont(x_prev, exc, inh)
    expected = x_prev + F.relu(exc - inh)
    assert torch.allclose(x_next, expected)

    # 2. Spiking binary mode: emits discrete binary pulses
    snn_bin = SpikingIntegrateAndFire(mode="spiking_binary", v_th=0.5)
    exc_var = exc.clone().requires_grad_(True)
    x_bin, mem_bin = snn_bin(x_prev, exc_var, inh)
    loss = x_bin.sum()
    loss.backward()
    # Check that gradients flow to excitatory input
    assert exc_var.grad is not None and exc_var.grad.abs().sum() > 0

    # 3. Spiking rate mode
    snn_rate = SpikingIntegrateAndFire(mode="spiking_rate", v_th=0.2, sub_steps=4)
    x_rate, _ = snn_rate(x_prev, exc, inh)
    assert x_rate.shape == x_prev.shape
    assert (x_rate >= x_prev).all()  # firing adds positive activity


def test_graph_circuit_factorization_exact_equivalence():
    """
    Validates Claim 3 / Appendix C.3:
    Arbitrary dense low-rank decomposition D E can be canonically factorized into
    non-negative circuit graphs H_e, H_i such that:
        (G_e - G_i) = (H_e^2 - H_i^2)[V] = (D_e - D_i) E' = D E identically.
    """
    torch.manual_seed(42)
    n = 64
    d = 16
    D = torch.randn(n, d)
    E = torch.randn(d, n)

    circuit = SparseLinearCircuit.from_dense_low_rank(D, E)
    v = torch.randn(4, n)

    # Propagate across two sparse steps (G = H^2)
    v_e, v_i = circuit(v)
    circuit_net = v_e - v_i

    # Direct dense product (D E) v
    dense_net = v @ (D @ E).T

    assert torch.allclose(circuit_net, dense_net, rtol=1e-4, atol=1e-4), (
        f"Max diff: {(circuit_net - dense_net).abs().max()}"
    )

    # Validate physical adjacency block matrix H_e and H_i
    H_e, H_i = circuit.get_circuit_graph()
    assert H_e.shape == (n + 2 * d, n + 2 * d)
    assert H_i.shape == (n + 2 * d, n + 2 * d)
    assert (H_e >= 0).all(), "H_e must have non-negative weights"
    assert (H_i >= 0).all(), "H_i must have non-negative weights"

    # Verify H^2 restriction to V
    H_e_sq = H_e @ H_e
    H_i_sq = H_i @ H_i
    G_e = H_e_sq[:n, :n]
    G_i = H_i_sq[:n, :n]
    net_G = G_e - G_i
    assert torch.allclose(net_G, D @ E, rtol=1e-4, atol=1e-4)


def test_graph_circuit_edge_sparsity():
    """Validates that the physical circuit contains O(nd) edges instead of n^2."""
    n = 256
    d = 16
    circuit = SparseLinearCircuit(n_neurons=n, dim_synaptic=d)
    edges = circuit.count_physical_edges()

    # Upper bound: s*n + n*s + n*s = 3 * (2d) * n = 6 * 16 * 256 = 24,576
    # Compared to dense n^2 = 65,536
    max_circuit_edges = 3 * (2 * d) * n
    assert edges <= max_circuit_edges
    assert edges < (n * n)


def test_explicit_synaptic_memory_hebbian_update():
    """
    Validates explicit synaptic memory matrix sigma_{t,l}:
    - Local Hebbian edge-reweighting update: sigma_{t,l} = (1 - u)*sigma_{t-1,l} + (y_{t,l-1} x_{t,l}^T) * G_s
    - Readout: a_{t,l}(j) = sum_i sigma_{t-1,l}(i, j) * x(i)
    """
    n = 32
    d = 8
    u = 0.1
    mem = SynapticMemory(n_neurons=n, dim_synaptic=d, damping=u, topology="hub_sparse")

    B = 2
    sigma_0 = mem.init_state(B, device=torch.device("cpu"))
    assert sigma_0.shape == (B, n, n)
    assert (sigma_0 == 0).all()

    # Step 1: Hebbian potentiation
    x_1 = torch.ones(B, n)
    y_0 = torch.full((B, n), 2.0)
    sigma_1 = mem.hebbian_update(sigma_0, x_post=x_1, y_pre=y_0)

    # Expected: 0 + (2.0 * 1.0) * G_s = 2.0 * G_s
    expected_sigma_1 = 2.0 * mem.G_s.unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(sigma_1, expected_sigma_1)

    # Step 2: Readout
    # a_1 = x_1 @ sigma_1
    a_1 = mem.readout(sigma_1, x_1)
    expected_a_1 = torch.bmm(x_1.unsqueeze(1), expected_sigma_1).squeeze(1)
    assert torch.allclose(a_1, expected_a_1)

    # Step 3: Damping on next update
    x_2 = torch.zeros(B, n)
    y_1 = torch.zeros(B, n)
    sigma_2 = mem.hebbian_update(sigma_1, x_post=x_2, y_pre=y_1)
    expected_sigma_2 = (1.0 - u) * sigma_1
    assert torch.allclose(sigma_2, expected_sigma_2)


def test_full_graph_bdh_forward_and_backward():
    """
    Validates end-to-end GraphBDH model:
    - Sequence processing (BPTT across tokens)
    - Gradient flow to all trainable parameters
    - Multiple threshold modes: continuous and spiking
    """
    for mode in ["continuous", "spiking_binary", "spiking_rate"]:
        config = GraphBDHConfig(
            n_layer=2,
            n_embd=16,
            n_neurons=64,
            vocab_size=32,
            damping=0.05,
            threshold_mode=mode,
            spiking_v_th=0.1,
            sub_steps=2,
            synaptic_topology="hub_sparse",
        )
        model = GraphBDH(config)

        B, T = 2, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx, targets)

        assert logits.shape == (B, T, config.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss) and not torch.isinf(loss)

        # Backward pass
        loss.backward()

        # Check gradients exist on all core parameters
        assert model.embed.weight.grad is not None
        assert model.lm_head.weight.grad is not None
        for layer in model.layers:
            assert layer.circuit_x.E_prime.grad is not None
            assert layer.circuit_x.D_e.grad is not None
            assert layer.circuit_x.D_i.grad is not None
            assert layer.circuit_y.E_prime.grad is not None


def test_graph_bdh_autoregressive_generation():
    """Validates state-preserving autoregressive generation."""
    config = GraphBDHConfig(
        n_layer=2,
        n_embd=16,
        n_neurons=64,
        vocab_size=32,
        threshold_mode="continuous",
    )
    model = GraphBDH(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)

    assert generated.shape == (1, 14)
    assert torch.equal(generated[:, :4], prompt)


def test_edge_and_sparsity_stats():
    """Validates edge counting and statistics reporting."""
    config = GraphBDHConfig(
        n_layer=3,
        n_embd=32,
        n_neurons=128,
        synaptic_topology="hub_sparse",
    )
    model = GraphBDH(config)
    stats = model.get_sparsity_and_edge_stats()

    assert stats["n_neurons"] == 128
    assert stats["dim_synaptic"] == 32
    assert stats["n_layer"] == 3
    assert stats["total_physical_circuit_edges"] > 0
    assert stats["total_physical_synapses"] > 0
    assert stats["edges_per_neuron"] > 0
