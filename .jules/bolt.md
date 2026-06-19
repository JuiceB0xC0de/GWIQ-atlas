## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.
## 2026-06-19 - Vectorized Sequence Length Masking
**Learning:** In batch processing of variable length sequences with left padding, creating boolean masks using Python `for` loops over batch dimensions introduces unnecessary Python interpreter overhead.
**Action:** Always use vectorized NumPy broadcasting (e.g. `np.arange(max_seq_len) >= (max_seq_len - np.array(seq_lens)[:, None])`) to compute boolean masks for sequence lengths instead of looping.
