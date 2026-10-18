# Copyright 2025 Pathway Technology, Inc. / Graph BDH Extension
"""
Graph BDH: Local Graph Dynamics Model with Sparse Synaptic State Updates
and Spiking/Threshold Primitives.

This module implements Phase 1 of the BDH graph formulation:
1. Graph Circuit Factorization (H_e, H_i):
   Replaces dense low-rank encoder/decoder matrices (E in R^{d x n}, D_x, D_y in R^{n x d})
   with a sparse linear circuit graph H in R^{(n+s) x (n+s)} containing m = O(nd) physical edges.
   Signal propagation occurs over two sparse steps (G = H^2), mapping directly to
   "communication by wire" topologies.
2. Explicit Synaptic Memory Matrix (sigma_{t,l}):
   Maintains the O(nd) physical synaptic matrix sigma_{t,l} rather than implicit low-rank
   mean-field attention (rho = E sigma). Implements local Hebbian edge-reweighting:
   sigma_{t,l} = (1 - u) sigma_{t-1,l} + (y_{t,l-1} x_{t,l}^T) * G_s
3. Spiking / Thresholding Primitives:
   Integrate-and-fire thresholding: x_{t,l} = x_{t,l-1} + (X_e - X_i)^+
   Supports continuous rectified thresholding as well as event-driven binary pulses
   and rate-coded spike trains using surrogate gradient backpropagation (fast sigmoid & atan).
"""

import dataclasses
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Spiking & Surrogate Gradient Primitives
# =============================================================================

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient for spiking threshold:
    Forward: Heaviside step function: S = 1 if (v - v_th) >= 0 else 0.
    Backward: Derivative of fast sigmoid:
        sigma'(x) = scale / (1 + |scale * x|)^2
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: float = 0.0, scale: float = 4.0):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        ctx.scale = scale
        return (v >= v_th).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        v_th = ctx.v_th
        scale = ctx.scale
        x = v - v_th
        # Fast sigmoid derivative
        grad_v = grad_output * scale / ((1.0 + (scale * x).abs()) ** 2)
        return grad_v, None, None


class ATanSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient for spiking threshold:
    Forward: Heaviside step function.
    Backward: Derivative of (1/pi)*arctan(scale * x) + 0.5:
        sigma'(x) = (scale / pi) / (1 + (scale * x)^2)
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: float = 0.0, scale: float = 4.0):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        ctx.scale = scale
        return (v >= v_th).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        v_th = ctx.v_th
        scale = ctx.scale
        x = v - v_th
        grad_v = grad_output * (scale / math.pi) / (1.0 + (scale * x) ** 2)
        return grad_v, None, None


class SpikingIntegrateAndFire(nn.Module):
    """
    Excitatory and Inhibitory Integrate-and-Fire Threshold Primitive:
    Implements: x_{t,l} = x_{t,l-1} + Threshold(X_e - X_i)
    
    Modes:
    - 'continuous': Rectified continuous thresholding (X_e - X_i)^+
    - 'spiking_binary': Event-driven binary pulse with surrogate gradient
    - 'spiking_rate': Multi-step rate-coded leaky integrate-and-fire spike train
    """
    def __init__(
        self,
        mode: str = "continuous",
        v_th: float = 0.1,
        leak: float = 0.9,
        surrogate: str = "fast_sigmoid",
        surrogate_scale: float = 4.0,
        sub_steps: int = 4,
    ):
        super().__init__()
        self.mode = mode
        self.v_th = v_th
        self.leak = leak
        self.surrogate = surrogate
        self.surrogate_scale = surrogate_scale
        self.sub_steps = sub_steps

    def _spike_fn(self, v: torch.Tensor, v_th: float) -> torch.Tensor:
        if self.surrogate == "fast_sigmoid":
            return FastSigmoidSurrogate.apply(v, v_th, self.surrogate_scale)
        elif self.surrogate == "atan":
            return ATanSurrogate.apply(v, v_th, self.surrogate_scale)
        else:
            raise ValueError(f"Unknown surrogate: {self.surrogate}")

    def forward(
        self,
        x_prev: torch.Tensor,
        exc_input: torch.Tensor,
        inh_input: torch.Tensor,
        membrane_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_prev: Previous activation tensor of shape (..., n)
            exc_input: Excitatory input X_e of shape (..., n)
            inh_input: Inhibitory input X_i of shape (..., n)
            membrane_state: Optional prior membrane potential (..., n)
        Returns:
            x_next: Updated activation tensor (..., n)
            membrane_state: Updated membrane potential (..., n)
        """
        net_current = exc_input - inh_input

        if self.mode == "continuous":
            # Direct rectified threshold: (X_e - X_i)^+
            delta = F.relu(net_current)
            x_next = x_prev + delta
            return x_next, membrane_state

        elif self.mode == "spiking_binary":
            # Event-driven single pulse with surrogate gradient
            v_mem = net_current
            if membrane_state is not None:
                v_mem = self.leak * membrane_state + net_current
            spikes = self._spike_fn(v_mem, self.v_th)
            # Soft reset
            new_v_mem = v_mem - self.v_th * spikes
            x_next = x_prev + spikes * self.v_th
            return x_next, new_v_mem

        elif self.mode == "spiking_rate":
            # Multi-step leaky integrate-and-fire simulation
            v_mem = torch.zeros_like(net_current) if membrane_state is None else membrane_state
            spike_accum = torch.zeros_like(net_current)
            sub_current = net_current / self.sub_steps

            for _ in range(self.sub_steps):
                v_mem = self.leak * v_mem + sub_current
                s = self._spike_fn(v_mem, self.v_th)
                v_mem = v_mem - self.v_th * s  # Soft reset
                spike_accum = spike_accum + s

            # Rate-coded activation output
            rate = spike_accum / float(self.sub_steps)
            x_next = x_prev + rate
            return x_next, v_mem
        else:
            raise ValueError(f"Unknown threshold mode: {self.mode}")


# =============================================================================
# 2. Graph Circuit Factorization (H_e, H_i) & 2-Step Signal Propagation (G = H^2)
# =============================================================================

class SparseLinearCircuit(nn.Module):
    """
    Sparse Linear Circuit Graph H in R^{(n+s) x (n+s)} with m = O(nd) physical edges.
    
    Replaces dense low-rank matrix product D E in R^{n x n} with a bipartite linear
    circuit topology between neuron nodes V (|V| = n) and synaptic landmark/hub nodes S (|S| = s).
    Typically s = 2d.
    
    Two-step signal propagation (G = H^2):
        Step 1 (V -> S): z_s = E' v in R^s (signals travel along input wires to hub nodes)
        Step 2 (S -> V): v_e = D_e z_s, v_i = D_i z_s in R^n (excitatory & inhibitory wire outputs)
        Net linear transformation: (G_e - G_i) v = (D_e - D_i) E' v.
    
    This preserves O(nd) parameters/edges while enabling direct mapping to
    "communication by wire" hardware topologies.
    """
    def __init__(
        self,
        n_neurons: int,
        dim_synaptic: int,
        s_hubs: Optional[int] = None,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.n = n_neurons
        self.d = dim_synaptic
        self.s = 2 * dim_synaptic if s_hubs is None else s_hubs

        # Non-negative parameter weights on physical wires:
        # E_prime: V -> S (input wires, shape (s, n))
        # D_e: S -> V (excitatory output wires, shape (n, s))
        # D_i: S -> V (inhibitory output wires, shape (n, s))
        self.E_prime = nn.Parameter(torch.empty(self.s, self.n))
        self.D_e = nn.Parameter(torch.empty(self.n, self.s))
        self.D_i = nn.Parameter(torch.empty(self.n, self.s))

        self.init_weights(init_std)

    def init_weights(self, std: float = 0.02):
        # Initialize with positive half-normal distribution to guarantee non-negative physical weights
        nn.init.normal_(self.E_prime, mean=std, std=std)
        nn.init.normal_(self.D_e, mean=std, std=std)
        nn.init.normal_(self.D_i, mean=std, std=std)
        with torch.no_grad():
            self.E_prime.clamp_(min=0.0)
            self.D_e.clamp_(min=0.0)
            self.D_i.clamp_(min=0.0)

    @classmethod
    def from_dense_low_rank(
        cls,
        D: torch.Tensor,
        E: torch.Tensor,
    ) -> "SparseLinearCircuit":
        """
        Exact canonical embedding of arbitrary real matrices D in R^{n x d} and E in R^{d x n}
        into non-negative circuit graphs H_e, H_i such that (G_e - G_i) = D E identically
        (as proved in Claim 3 & Appendix C.3 of the BDH paper).
        """
        n, d = D.shape
        circuit = cls(n_neurons=n, dim_synaptic=d, s_hubs=2 * d)
        with torch.no_grad():
            # E_prime in R^{2d x n}: [(E)^+; (-E)^+]
            E_pos = F.relu(E)
            E_neg = F.relu(-E)
            circuit.E_prime.copy_(torch.cat([E_pos, E_neg], dim=0))

            # D_e, D_i in R^{n x 2d}:
            # (D_e - D_i) E_prime = [D_e1, D_e2] [E_pos; E_neg] - [D_i1, D_i2] [E_pos; E_neg]
            #                     = (D_e1 - D_i1) E_pos + (D_e2 - D_i2) E_neg = D(E_pos - E_neg) = DE
            # We set: D_e1 = (D)^+, D_i1 = (-D)^+, D_e2 = (-D)^+, D_i2 = (D)^+
            D_pos = F.relu(D)
            D_neg = F.relu(-D)
            circuit.D_e.copy_(torch.cat([D_pos, D_neg], dim=1))
            circuit.D_i.copy_(torch.cat([D_neg, D_pos], dim=1))
        return circuit

    def get_circuit_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs explicit (n+s) x (n+s) adjacency matrices for H_e and H_i:
            H_e = [[0, D_e], [E', 0]],  H_i = [[0, D_i], [E', 0]]
        """
        n, s = self.n, self.s
        device, dtype = self.E_prime.device, self.E_prime.dtype

        # H_e block matrix
        H_e = torch.zeros(n + s, n + s, device=device, dtype=dtype)
        H_e[:n, n:] = F.relu(self.D_e)
        H_e[n:, :n] = F.relu(self.E_prime)

        # H_i block matrix
        H_i = torch.zeros(n + s, n + s, device=device, dtype=dtype)
        H_i[:n, n:] = F.relu(self.D_i)
        H_i[n:, :n] = F.relu(self.E_prime)

        return H_e, H_i

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-step sparse signal propagation (v -> H^2 v):
        Step 1 (V -> S): z_s = E' v
        Step 2 (S -> V): v_e = D_e z_s, v_i = D_i z_s
        
        Args:
            v: Input neuron signal of shape (..., n)
        Returns:
            (v_e, v_i): Excitatory and inhibitory signals of shape (..., n)
        """
        # Ensure non-negative wire weights during forward pass
        E_wire = F.relu(self.E_prime)  # (s, n)
        D_e_wire = F.relu(self.D_e)    # (n, s)
        D_i_wire = F.relu(self.D_i)    # (n, s)

        # Step 1: Signals propagate across input wires V -> S
        # v: (..., n) -> z_s: (..., s)
        z_s = F.linear(v, E_wire)

        # Step 2: Signals propagate across output wires S -> V
        # z_s: (..., s) -> v_e, v_i: (..., n)
        v_e = F.linear(z_s, D_e_wire)
        v_i = F.linear(z_s, D_i_wire)

        return v_e, v_i

    def count_physical_edges(self, threshold: float = 1e-5) -> int:
        """Returns total number of active non-zero physical wires (edges) in the circuit."""
        with torch.no_grad():
            nnz_E = (F.relu(self.E_prime) > threshold).sum().item()
            nnz_De = (F.relu(self.D_e) > threshold).sum().item()
            nnz_Di = (F.relu(self.D_i) > threshold).sum().item()
            return int(nnz_E + nnz_De + nnz_Di)


# =============================================================================
# 3. Explicit Synaptic Memory Matrix (sigma_{t,l}) & Local Hebbian Updates
# =============================================================================

class SynapticMemory(nn.Module):
    """
    Explicit Physical Synaptic Memory Matrix sigma_{t,l} in R^{n x n}.
    
    Replaces implicit low-rank tensor attention rho_{t,l} = E sigma_{t,l} with the
    O(nd) sparse physical synaptic matrix sigma_{t,l}.
    
    Implements local Hebbian edge-reweighting dynamics:
        sigma_{t,l} = (1 - u) sigma_{t-1,l} + (y_{t,l-1} x_{t,l}^T) * G_s
    where:
        - G_s in {0, 1}^{n x n} is the physical synaptic connectivity topology (O(nd) edges).
        - u is the damping / decay rate (0 < u < 1).
        - Readout across physical synapses:
            a_{t,l}(j) = sum_i sigma_{t-1,l}(i, j) * x_{t,l}(i)  [Table 1: X(i), sigma_l(i,j) -> A(j)]
    """
    def __init__(
        self,
        n_neurons: int,
        dim_synaptic: int,
        damping: float = 0.05,
        topology: str = "hub_sparse",
        learnable_damping: bool = False,
    ):
        super().__init__()
        self.n = n_neurons
        self.d = dim_synaptic
        self.topology = topology

        # Physical synaptic mask G_s: 1 if physical synapse exists, 0 otherwise
        G_s = self._build_synaptic_mask(n_neurons, dim_synaptic, topology)
        self.register_buffer("G_s", G_s)

        # Synaptic decay factor u
        if learnable_damping:
            # Parametrized via sigmoid for stable bounded damping in (0, 1)
            self.u_raw = nn.Parameter(torch.full((n_neurons,), math.logit(damping)))
        else:
            self.register_buffer("damping", torch.tensor(damping))
            self.u_raw = None

    def _build_synaptic_mask(self, n: int, d: int, topology: str) -> torch.Tensor:
        """
        Constructs O(nd) sparse physical synaptic connectivity mask G_s.
        """
        if topology == "dense":
            # Baseline dense all-to-all connectivity (n^2 edges)
            return torch.ones(n, n, dtype=torch.float32)

        elif topology == "hub_sparse":
            # Canonical hub-landmark topology from Claim 4 / Claim 9 (Appendix C.4):
            # 2d landmark hub columns connect to all neurons, giving 2dn = O(nd) physical synapses
            mask = torch.zeros(n, n, dtype=torch.float32)
            n_hubs = min(2 * d, n)
            mask[:, :n_hubs] = 1.0
            return mask

        elif topology == "power_law_sparse":
            # Scale-free / modular sparse topology with O(nd) edges
            mask = torch.zeros(n, n, dtype=torch.float32)
            k = min(4 * d, n)  # average degree ~ 4d
            for i in range(n):
                # Local band connections + random shortcuts
                local_indices = [(i + offset) % n for offset in range(-k // 4, k // 4 + 1)]
                mask[i, local_indices] = 1.0
                # Heavy-tailed hub shortcuts
                hubs = torch.randint(0, min(2 * d, n), (k // 2,))
                mask[i, hubs] = 1.0
            return mask
        else:
            raise ValueError(f"Unknown synaptic topology: {topology}")

    def get_damping(self) -> torch.Tensor:
        if self.u_raw is not None:
            return torch.sigmoid(self.u_raw)
        return self.damping

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes empty synaptic state sigma_0 = 0 in shape (B, n, n)."""
        return torch.zeros(batch_size, self.n, self.n, device=device)

    def readout(self, sigma_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Synaptic transmission: propagates activation x across physical synapses sigma_{t-1,l}.
        a_{t,l}(j) = sum_i sigma_{t-1,l}(i, j) * x(i)
        
        Args:
            sigma_prev: Physical synaptic state of shape (B, n, n)
            x: Current neuron activation x_{t,l} of shape (B, n)
        Returns:
            a: Retrieved associative activation of shape (B, n)
        """
        # x is (B, 1, n), sigma_prev is (B, n, n) -> a = x @ sigma_prev -> (B, n)
        # Table 1: X(i), sigma_l(i, j) -> A(j)
        a = torch.bmm(x.unsqueeze(1), sigma_prev).squeeze(1)
        return a

    def hebbian_update(
        self,
        sigma_prev: torch.Tensor,
        x_post: torch.Tensor,
        y_pre: torch.Tensor,
    ) -> torch.Tensor:
        """
        Local Hebbian edge-reweighting update:
            sigma_{t,l} = (1 - u) * sigma_{t-1,l} + (y_{t,l-1} * x_{t,l}^T) * G_s
        
        Args:
            sigma_prev: Previous synaptic matrix (B, n, n)
            x_post: Post-synaptic activation x_{t,l} of shape (B, n)
            y_pre: Pre-synaptic activation y_{t,l-1} of shape (B, n)
        Returns:
            sigma_next: Updated synaptic matrix (B, n, n)
        """
        u = self.get_damping()  # scalar or (n,)
        decay = 1.0 - u

        # Outer product of pre-synaptic and post-synaptic activity:
        # y_pre is (B, n, 1), x_post is (B, 1, n) -> outer product is (B, n, n)
        hebbian_delta = torch.bmm(y_pre.unsqueeze(2), x_post.unsqueeze(1))

        # Restrict potentiation strictly to physical synaptic edges G_s
        hebbian_delta = hebbian_delta * self.G_s

        if decay.ndim == 1:
            decay = decay.view(1, 1, -1)  # Broadcast across columns

        sigma_next = decay * sigma_prev + hebbian_delta
        return sigma_next

    def count_physical_synapses(self) -> int:
        """Returns the number of active physical synapses in G_s."""
        return int(self.G_s.sum().item())


# =============================================================================
# 4. Graph BDH Layer & Architecture
# =============================================================================

@dataclasses.dataclass
class GraphBDHConfig:
    n_layer: int = 4
    n_embd: int = 64                  # Low-rank dimension d
    n_neurons: int = 256              # Total neurons n (where n >> d)
    vocab_size: int = 256
    damping: float = 0.05             # Synaptic decay rate u
    threshold_mode: str = "continuous"  # 'continuous', 'spiking_binary', 'spiking_rate'
    spiking_v_th: float = 0.1         # Membrane threshold
    spiking_leak: float = 0.9         # Membrane leak factor
    surrogate: str = "fast_sigmoid"   # 'fast_sigmoid' or 'atan'
    surrogate_scale: float = 4.0      # Surrogate gradient sharpness
    sub_steps: int = 4                # Sub-steps for rate-coded spiking
    synaptic_topology: str = "hub_sparse"  # 'dense', 'hub_sparse', 'power_law_sparse'
    dropout: float = 0.0


class GraphBDHLayer(nn.Module):
    """
    Single layer of Graph BDH executing local graph dynamics:
    1. Linear Circuit propagation G_y = H_y^2 on synaptic readout a_{t,l} = sigma_{t-1,l} x_{t,l}.
    2. Integrate-and-fire thresholding and replicator dynamics:
       y_{t,l} = Threshold(Y_e - Y_i) * x_{t,l}
    3. Linear Circuit propagation G_x = H_x^2 on y_{t,l}:
       X_e, X_i = Circuit_x(y_{t,l})
    4. State accumulation:
       x_{t,l} = x_{t,l-1} + Threshold(X_e - X_i)
    5. Local Hebbian synaptic update:
       sigma_{t,l} = (1 - u) sigma_{t-1,l} + (y_{t,l-1} x_{t,l}^T) * G_s
    """
    def __init__(self, config: GraphBDHConfig):
        super().__init__()
        self.config = config
        n = config.n_neurons
        d = config.n_embd

        # Sparse linear circuits H_x and H_y (2-step signal propagation G = H^2)
        self.circuit_x = SparseLinearCircuit(n_neurons=n, dim_synaptic=d)
        self.circuit_y = SparseLinearCircuit(n_neurons=n, dim_synaptic=d)

        # Explicit physical synaptic memory matrix sigma_{t,l}
        self.synaptic_memory = SynapticMemory(
            n_neurons=n,
            dim_synaptic=d,
            damping=config.damping,
            topology=config.synaptic_topology,
        )

        # Spiking / integrate-and-fire thresholding primitives
        self.threshold_x = SpikingIntegrateAndFire(
            mode=config.threshold_mode,
            v_th=config.spiking_v_th,
            leak=config.spiking_leak,
            surrogate=config.surrogate,
            surrogate_scale=config.surrogate_scale,
            sub_steps=config.sub_steps,
        )
        self.threshold_y = SpikingIntegrateAndFire(
            mode=config.threshold_mode,
            v_th=config.spiking_v_th,
            leak=config.spiking_leak,
            surrogate=config.surrogate,
            surrogate_scale=config.surrogate_scale,
            sub_steps=config.sub_steps,
        )

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def step(
        self,
        x_prev_layer: torch.Tensor,
        y_prev_layer: torch.Tensor,
        x_prev_time: torch.Tensor,
        sigma_prev_time: torch.Tensor,
        mem_x: Optional[torch.Tensor] = None,
        mem_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Executes one recurrent step of local graph dynamics for current token t:
        Args:
            x_prev_layer: Input activation from previous layer x_{t, l-1} of shape (B, n)
            y_prev_layer: Output activation from previous layer y_{t, l-1} of shape (B, n)
            x_prev_time: Activation from previous time step x_{t-1, l} of shape (B, n)
            sigma_prev_time: Synaptic matrix from previous time step sigma_{t-1, l} of shape (B, n, n)
            mem_x: Optional prior membrane potential for neuron x
            mem_y: Optional prior membrane potential for neuron y
        Returns:
            (x_curr, y_curr, sigma_curr, new_mem_x, new_mem_y)
        """
        # 1. Readout from physical synaptic memory:
        # a_{t,l} = sigma_{t-1,l} * x_{t,l-1}
        a = self.synaptic_memory.readout(sigma_prev_time, x_prev_layer)

        # 2. Wire communication over circuit G_y = H_y^2
        # (Y_e, Y_i) = Circuit_y(a)
        Y_e, Y_i = self.circuit_y(a)

        # 3. Integrate-and-fire thresholding on Y followed by replicator competition:
        # y_{t,l} = Threshold(Y_e - Y_i) * x_{t, l-1}
        y_thresh, new_mem_y = self.threshold_y(
            x_prev=torch.zeros_like(x_prev_layer),
            exc_input=Y_e,
            inh_input=Y_i,
            membrane_state=mem_y,
        )
        y_curr = y_thresh * x_prev_layer  # Replicator dynamics: elementwise multiplication
        y_curr = self.drop(y_curr)

        # 4. Wire communication over circuit G_x = H_x^2
        # (X_e, X_i) = Circuit_x(y_curr)
        X_e, X_i = self.circuit_x(y_curr)

        # 5. Integrate-and-fire thresholding on X:
        # x_{t,l} = x_{t,l-1} + Threshold(X_e - X_i)
        x_curr, new_mem_x = self.threshold_x(
            x_prev=x_prev_layer,
            exc_input=X_e,
            inh_input=X_i,
            membrane_state=mem_x,
        )

        # 6. Local Hebbian edge-reweighting on physical synapses:
        # sigma_{t,l} = (1 - u) sigma_{t-1,l} + (y_{t,l-1} x_{t,l}^T) * G_s
        sigma_curr = self.synaptic_memory.hebbian_update(
            sigma_prev=sigma_prev_time,
            x_post=x_curr,
            y_pre=y_prev_layer,
        )

        return x_curr, y_curr, sigma_curr, new_mem_x, new_mem_y


class GraphBDH(nn.Module):
    """
    Graph BDH: End-to-end Language Model based on Explicit Local Graph Dynamics.
    """
    def __init__(self, config: GraphBDHConfig):
        super().__init__()
        self.config = config
        n = config.n_neurons

        # Embedding: maps token IDs to initial neuron activations x_{t, 0} in R^n
        self.embed = nn.Embedding(config.vocab_size, n)

        # Layers of local graph dynamics
        self.layers = nn.ModuleList([
            GraphBDHLayer(config) for _ in range(config.n_layer)
        ])

        # Readout head: maps final neuron activations x_{t, L} in R^n to vocabulary logits
        self.lm_head = nn.Linear(n, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def init_state(self, batch_size: int, device: torch.device):
        """Initializes state tuples for all layers (synaptic matrices and membrane potentials)."""
        states = []
        for layer in self.layers:
            sigma = layer.synaptic_memory.init_state(batch_size, device)
            states.append({
                "sigma": sigma,
                "mem_x": None,
                "mem_y": None,
                "x_prev": torch.zeros(batch_size, self.config.n_neurons, device=device),
            })
        return states

    def forward_step(
        self,
        token_id: torch.Tensor,
        states: list,
    ) -> Tuple[torch.Tensor, list]:
        """
        Executes one forward step for a single token at time step t across all layers.
        Args:
            token_id: LongTensor of shape (B,) or (B, 1)
            states: List of state dictionaries per layer
        Returns:
            logits: (B, vocab_size)
            new_states: Updated states per layer
        """
        if token_id.ndim == 2:
            token_id = token_id.squeeze(1)

        # Initial neuron input: positive rectified embedding
        x = F.relu(self.embed(token_id))
        y = x.clone()

        new_states = []
        for l_idx, layer in enumerate(self.layers):
            st = states[l_idx]
            x_next, y_next, sigma_next, mem_x_next, mem_y_next = layer.step(
                x_prev_layer=x,
                y_prev_layer=y,
                x_prev_time=st["x_prev"],
                sigma_prev_time=st["sigma"],
                mem_x=st["mem_x"],
                mem_y=st["mem_y"],
            )
            new_states.append({
                "sigma": sigma_next,
                "mem_x": mem_x_next,
                "mem_y": mem_y_next,
                "x_prev": x_next,
            })
            x = x_next
            y = y_next

        logits = self.lm_head(x)
        return logits, new_states

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Processes a sequence of tokens of shape (B, T) by unrolling the local graph dynamics.
        Enables end-to-end backpropagation through time (BPTT) using surrogate gradients.
        """
        B, T = idx.shape
        device = idx.device
        states = self.init_state(B, device)

        all_logits = []
        for t in range(T):
            token_t = idx[:, t]
            logits_t, states = self.forward_step(token_t, states)
            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive token generation maintaining explicit synaptic state.
        """
        B, T = idx.shape
        device = idx.device
        states = self.init_state(B, device)

        # Warm up states on prompt context
        for t in range(T):
            token_t = idx[:, t]
            logits, states = self.forward_step(token_t, states)

        # Generate tokens
        curr_token = idx[:, -1]
        out_tokens = [idx]

        for _ in range(max_new_tokens):
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out_tokens.append(next_token)
            logits, states = self.forward_step(next_token.squeeze(1), states)

        return torch.cat(out_tokens, dim=1)

    def get_sparsity_and_edge_stats(self) -> dict:
        """Returns statistics on physical edges, synaptic density, and parameter counts."""
        total_circuit_edges = 0
        total_synapses = 0
        for layer in self.layers:
            total_circuit_edges += layer.circuit_x.count_physical_edges()
            total_circuit_edges += layer.circuit_y.count_physical_edges()
            total_synapses += layer.synaptic_memory.count_physical_synapses()

        return {
            "n_neurons": self.config.n_neurons,
            "dim_synaptic": self.config.n_embd,
            "n_layer": self.config.n_layer,
            "total_physical_circuit_edges": total_circuit_edges,
            "total_physical_synapses": total_synapses,
            "edges_per_neuron": (total_circuit_edges + total_synapses) / self.config.n_neurons,
        }
