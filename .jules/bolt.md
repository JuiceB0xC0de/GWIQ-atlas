## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2024-05-16 - [Vectorized Sequence Length Masking]
**Learning:** In the `qwip-atlas` codebase, creating boolean masks for variable-length sequences using a Python `for` loop (e.g., iterating over `enumerate(seq_lens)`) causes significant performance overhead, especially for large batch sizes. This can dramatically increase extraction times for operations calculating sequence means.
**Action:** When creating boolean masks for batched sequences with variable lengths (e.g., left-padded token masks), always use vectorized numpy broadcasting operations like `np.arange(max_seq_len) >= (max_seq_len - np.array(seq_lens)[:, None])`. This replaces the loop with heavily optimized C operations.
