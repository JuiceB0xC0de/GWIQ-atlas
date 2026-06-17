## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2024-05-24 - [Slice Tensors Before Expensive Operations]
**Learning:** When processing PyTorch tensors in `qwip-atlas` extraction pipelines, always slice the tensor down to the required subset (e.g., selecting the last token) *before* applying computationally expensive operations like activation functions or reshapes to prevent redundant O(batch_size * seq_len) calculations.
**Action:** In `_last_token_components`, refactored to extract the slice `t[batch_idx, sl][-1]` before calling `act_fn()` and `_safe_per_head_last()`. This drastically reduced computation time from processing entire sequences to processing single tokens.
