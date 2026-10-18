# Copyright 2025 Pathway Technology, Inc. / Graph BDH Neuromorphic Profiler
"""
Neuromorphic Hardware Simulator & Energy Profiler for Graph BDH.

Simulates event-driven execution on neuromorphic hardware architectures (e.g. Intel Loihi 1/2,
IBM TrueNorth, SpiNNaker) without physical hardware.

Metrics Computed:
1. Spike Sparsity & Activity Ratio: Fraction of neurons active per token.
2. SynOps (Synaptic Operations):
   - Wires V -> S and S -> V traversed by discrete pulses in circuits H_x and H_y.
   - Physical synaptic transmissions through sigma_{t,l} and G_s.
   - Hebbian plasticity events on physical synapses.
3. Energy Benchmarking:
   - Neuromorphic Energy: E_neuro = (SynOps * E_synop) + (ActiveNeurons * E_neuron) + (P_static * latency)
     Standard literature references:
       - Intel Loihi 1: ~23.6 pJ / SynOp (Blouw et al., 2019)
       - Intel Loihi 2: ~1.5 - 5.0 pJ / SynOp (Davies et al., 2021)
       - Neuromorphic Neuron update: ~50 - 100 pJ
   - Dense GPU Equivalent: E_gpu = MACs * E_mac (where E_mac ~ 20-50 pJ including SRAM/DRAM access).
   - Energy Efficiency Multiplier: E_gpu / E_neuro
"""

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from graph_bdh import GraphBDH, GraphBDHConfig


@dataclasses.dataclass
class NeuromorphicHardwareSpecs:
    name: str = "Loihi-2-Class"
    energy_per_synop_pj: float = 3.5      # Picojoules per Synaptic Operation (SynOp)
    energy_per_neuron_update_pj: float = 40.0  # Picojoules per active neuron update
    static_power_mw: float = 15.0         # Static / leakage power in milliwatts
    clock_frequency_mhz: float = 100.0    # Operating frequency
    # Dense GPU baseline comparison specs:
    gpu_energy_per_mac_pj: float = 35.0   # GPU FP16 MAC energy + SRAM/DRAM fetch overhead


@dataclasses.dataclass
class TokenNeuromorphicProfile:
    token_idx: int
    n_neurons: int
    active_neurons: int
    activation_sparsity_pct: float
    circuit_synops: int
    memory_readout_synops: int
    plasticity_synops: int
    total_synops: int
    dense_equivalent_macs: int
    estimated_neuromorphic_energy_nj: float
    estimated_gpu_energy_nj: float
    energy_efficiency_multiplier: float


class NeuromorphicProfiler:
    """
    Hooks into Graph BDH inference to profile spike events and synaptic operations (SynOps).
    """
    def __init__(
        self,
        model: GraphBDH,
        specs: Optional[NeuromorphicHardwareSpecs] = None,
    ):
        self.model = model
        self.specs = specs or NeuromorphicHardwareSpecs()

    def profile_generation(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 5,
    ) -> Tuple[torch.Tensor, List[TokenNeuromorphicProfile]]:
        """
        Profiles autoregressive generation token-by-token.
        """
        self.model.eval()
        device = prompt_tokens.device
        B = prompt_tokens.shape[0]
        assert B == 1, "Profiler currently supports single-sequence inference (B=1)."

        states = self.model.init_state(B, device)
        profiles: List[TokenNeuromorphicProfile] = []

        # Warm up on prompt
        for t in range(prompt_tokens.shape[1]):
            token_t = prompt_tokens[:, t]
            logits, states = self.model.forward_step(token_t, states)

        # Generate and profile new tokens
        curr_token = prompt_tokens[:, -1]
        out_tokens = [prompt_tokens]

        for step in range(max_new_tokens):
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out_tokens.append(next_token)

            # Profile single step
            profile, new_states, next_logits = self._profile_single_step(
                next_token.squeeze(1), states, token_idx=step
            )
            profiles.append(profile)
            states = new_states
            logits = next_logits

        generated = torch.cat(out_tokens, dim=1)
        return generated, profiles

    def _profile_single_step(
        self,
        token_id: torch.Tensor,
        states: list,
        token_idx: int,
    ) -> Tuple[TokenNeuromorphicProfile, list, torch.Tensor]:
        """
        Profiles physical wire and synapse traversals for a single token.
        """
        cfg = self.model.config
        n = cfg.n_neurons

        # Initial embedding
        x = F.relu(self.model.embed(token_id))
        y = x.clone()

        total_active_neurons = 0
        total_circuit_synops = 0
        total_memory_synops = 0
        total_plasticity_synops = 0
        new_states = []

        for l_idx, layer in enumerate(self.model.layers):
            st = states[l_idx]
            sigma_prev = st["sigma"]

            # 1. Synaptic Memory Readout: a = x @ sigma_prev
            # Active neurons in x trigger transmissions along incoming non-zero synapses
            x_active_mask = (x > 1e-4).squeeze(0)  # (n,)
            active_synapses_in_sigma = (sigma_prev.squeeze(0).abs() > 1e-6) & (layer.synaptic_memory.G_s > 0)
            # SynOps for readout: sum of active pre-synaptic neurons times their outgoing connected synapses
            readout_synops = active_synapses_in_sigma[:, x_active_mask].sum().item()
            total_memory_synops += int(readout_synops)

            # Synaptic readout
            a = layer.synaptic_memory.readout(sigma_prev, x)

            # 2. Circuit Y (G_y = H_y^2): v -> z_s = E' a -> (D_e z_s, D_i z_s)
            a_active_mask = (a > 1e-4).squeeze(0)
            E_wire_y = F.relu(layer.circuit_y.E_prime)  # (s, n)
            # Step 1: active neurons in 'a' travel to synaptic hubs S
            step1_synops_y = (E_wire_y[:, a_active_mask] > 1e-5).sum().item()

            z_s_y = F.linear(a, E_wire_y)
            z_s_active_y = (z_s_y > 1e-4).squeeze(0)

            # Step 2: active hubs in S travel along D_e and D_i to neurons V
            D_e_y = F.relu(layer.circuit_y.D_e)
            D_i_y = F.relu(layer.circuit_y.D_i)
            step2_synops_y = (
                (D_e_y[:, z_s_active_y] > 1e-5).sum().item()
                + (D_i_y[:, z_s_active_y] > 1e-5).sum().item()
            )
            total_circuit_synops += int(step1_synops_y + step2_synops_y)

            # Execute forward
            Y_e = F.linear(z_s_y, D_e_y)
            Y_i = F.linear(z_s_y, D_i_y)

            y_thresh, new_mem_y = layer.threshold_y(
                x_prev=torch.zeros_like(x),
                exc_input=Y_e,
                inh_input=Y_i,
                membrane_state=st["mem_y"],
            )
            y_curr = y_thresh * x

            # 3. Circuit X (G_x = H_x^2): y_curr -> z_s_x -> (X_e, X_i)
            y_active_mask = (y_curr > 1e-4).squeeze(0)
            E_wire_x = F.relu(layer.circuit_x.E_prime)
            step1_synops_x = (E_wire_x[:, y_active_mask] > 1e-5).sum().item()

            z_s_x = F.linear(y_curr, E_wire_x)
            z_s_active_x = (z_s_x > 1e-4).squeeze(0)

            D_e_x = F.relu(layer.circuit_x.D_e)
            D_i_x = F.relu(layer.circuit_x.D_i)
            step2_synops_x = (
                (D_e_x[:, z_s_active_x] > 1e-5).sum().item()
                + (D_i_x[:, z_s_active_x] > 1e-5).sum().item()
            )
            total_circuit_synops += int(step1_synops_x + step2_synops_x)

            X_e = F.linear(z_s_x, D_e_x)
            X_i = F.linear(z_s_x, D_i_x)

            x_curr, new_mem_x = layer.threshold_x(
                x_prev=x,
                exc_input=X_e,
                inh_input=X_i,
                membrane_state=st["mem_x"],
            )

            # 4. Plasticity SynOps: pairs of active (x_post, y_pre) along physical edges G_s
            x_post_active = (x_curr > 1e-4).squeeze(0)
            y_pre_active = (y > 1e-4).squeeze(0)
            plastic_events = (
                layer.synaptic_memory.G_s[y_pre_active, :][:, x_post_active].sum().item()
            )
            total_plasticity_synops += int(plastic_events)

            sigma_next = layer.synaptic_memory.hebbian_update(
                sigma_prev=sigma_prev,
                x_post=x_curr,
                y_pre=y,
            )

            new_states.append({
                "sigma": sigma_next,
                "mem_x": new_mem_x,
                "mem_y": new_mem_y,
                "x_prev": x_curr,
            })

            layer_active = int(x_post_active.sum().item())
            total_active_neurons += layer_active
            x = x_curr
            y = y_curr

        logits = self.model.lm_head(x)

        # Totals and Energy Calculations
        total_synops = total_circuit_synops + total_memory_synops + total_plasticity_synops
        total_possible_neurons = cfg.n_neurons * cfg.n_layer
        sparsity_pct = (1.0 - (total_active_neurons / max(1, total_possible_neurons))) * 100.0

        # Dense GPU baseline: dense MACs per token is roughly total model parameter count
        dense_macs = sum(p.numel() for p in self.model.parameters())

        # Energy in nanojoules (nJ)
        e_synops_pj = total_synops * self.specs.energy_per_synop_pj
        e_neurons_pj = total_active_neurons * self.specs.energy_per_neuron_update_pj
        total_neuro_energy_nj = (e_synops_pj + e_neurons_pj) / 1000.0

        total_gpu_energy_nj = (dense_macs * self.specs.gpu_energy_per_mac_pj) / 1000.0
        multiplier = (
            total_gpu_energy_nj / total_neuro_energy_nj if total_neuro_energy_nj > 0 else 1.0
        )

        profile = TokenNeuromorphicProfile(
            token_idx=token_idx,
            n_neurons=total_possible_neurons,
            active_neurons=total_active_neurons,
            activation_sparsity_pct=sparsity_pct,
            circuit_synops=total_circuit_synops,
            memory_readout_synops=total_memory_synops,
            plasticity_synops=total_plasticity_synops,
            total_synops=total_synops,
            dense_equivalent_macs=dense_macs,
            estimated_neuromorphic_energy_nj=total_neuro_energy_nj,
            estimated_gpu_energy_nj=total_gpu_energy_nj,
            energy_efficiency_multiplier=multiplier,
        )

        return profile, new_states, logits

    def print_summary(self, profiles: List[TokenNeuromorphicProfile]):
        """Prints a human-readable summary table of neuromorphic profiling results."""
        if not profiles:
            print("No profiles recorded.")
            return

        avg_sparsity = sum(p.activation_sparsity_pct for p in profiles) / len(profiles)
        avg_synops = sum(p.total_synops for p in profiles) / len(profiles)
        avg_neuro_nj = sum(p.estimated_neuromorphic_energy_nj for p in profiles) / len(profiles)
        avg_gpu_nj = sum(p.estimated_gpu_energy_nj for p in profiles) / len(profiles)
        avg_mult = sum(p.energy_efficiency_multiplier for p in profiles) / len(profiles)
        dense_macs = profiles[0].dense_equivalent_macs

        print("\n================ Neuromorphic Hardware Simulation Summary ================")
        print(f"Target Hardware Profile: {self.specs.name}")
        print(f"Energy per SynOp: {self.specs.energy_per_synop_pj} pJ | Energy per Active Neuron: {self.specs.energy_per_neuron_update_pj} pJ")
        print(f"Tokens Profiled: {len(profiles)}")
        print("--------------------------------------------------------------------------")
        print(f"Average Neuronal Sparsity:       {avg_sparsity:6.2f}% (inactive quiescent neurons)")
        print(f"Average Synaptic Operations:     {avg_synops:10.1f} SynOps / token")
        print(f"Dense GPU Baseline MACs:         {dense_macs:10,d} MACs / token")
        print("--------------------------------------------------------------------------")
        print(f"Estimated Neuromorphic Energy:   {avg_neuro_nj:8.3f} nJ / token ({avg_neuro_nj * 1e-3:.4f} uJ/token)")
        print(f"Estimated Dense GPU Energy:      {avg_gpu_nj:8.3f} nJ / token ({avg_gpu_nj * 1e-3:.4f} uJ/token)")
        print(f"Neuromorphic Efficiency Factor:  {avg_mult:8.1f}x lower energy")
        print("==========================================================================\n")
