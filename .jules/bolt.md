## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2024-05-18 - [Vectorized Slicing for Left-Padded Tensors]
**Learning:** In PyTorch codebases that use `padding_side="left"` for tokenization, the last token of any sequence is always found at index `-1`. Slicing `[:, -1]` across a batch avoids the need for per-item loops over individual sequence lengths.
**Action:** Always verify `padding_side` when extracting the last token. If left padded, vectorize the extraction with `tensor[:, -1]` before applying heavy computations like activation functions.
