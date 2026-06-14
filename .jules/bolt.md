## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2024-06-25 - [Sequence Processing Optimization]
**Learning:** In the `qwip-atlas` codebase's extraction pipelines (`compliance_behaviour.py`), applying computationally expensive PyTorch operations (like `silu` activations or large reshapes) to the entire sequence tensor `[batch_size, seq_len, hidden_dim]` before slicing down to the target tokens results in massive redundant O(batch_size * seq_len) calculation overheads.
**Action:** Always slice PyTorch tensors down to the required subsets (e.g., extracting the last token) *before* applying activations, reshapes, or other non-linearities.
