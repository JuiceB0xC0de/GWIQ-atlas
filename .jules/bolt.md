## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2026-06-16 - [Vectorized Boolean Mask Creation]
**Learning:** In the `qwip-atlas` codebase, creating boolean masks using a Python `for` loop over the batch dimension (e.g. `for i, length in enumerate(seq_lens): mask[i, -length:] = True`) is a bottleneck, as it creates overhead when extracting batches, even for small batch sizes.
**Action:** Replace Python loops for boolean mask generation with vectorized numpy broadcasting operations like `np.arange(max_seq_len) >= (max_seq_len - seq_lens_arr[:, None])`.
