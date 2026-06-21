## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.
## 2024-06-21 - [Vectorizing Sequence Length Masks]
**Learning:** In the `qwip-atlas` codebase, constructing boolean masks for variable sequence lengths in batches using Python `for` loops inside the hot extraction path creates significant performance overhead due to the high frequency of calls during the census phase.
**Action:** Use vectorized numpy broadcasting operations like `np.arange(max_seq_len) >= (max_seq_len - np.array(seq_lens)[:, None])` instead of Python `for` loops for mask generation to leverage efficient C backends and avoid CPU overhead.
