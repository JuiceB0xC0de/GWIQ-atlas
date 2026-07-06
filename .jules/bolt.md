## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2024-07-06 - [Vectorized Batch Extraction]
**Learning:** Extracting token representations one batch element at a time using `.cpu().float().numpy()` and computationally expensive operations (like `act_fn` and reshaping) causes huge performance bottlenecks due to excessive D2H synchronization, unneeded ops on full sequence lengths, and redundant function calls.
**Action:** When extracting batched token representation from PyTorch models, apply indexing logic to slice down to target tokens (`[:, -1]`) *before* processing `act_fn` or `.cpu().numpy()`, allowing NumPy to handle processing once over the whole batch instead of iterating over individual batch elements.
