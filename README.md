# Autokernel Experiment

Experiments are explored in three dimensions: coding agent, target computing, and set of operators.

## Project Structure

```
autokernel_output/
├── README.md
└── <coding_agent>/                         # e.g., sonnet
    └── <target_device>/                    # e.g., h100, rtx_5070
        ├── nvidia_*.json                   # Device specifications
        └── workspace_<model>/              # e.g., workspace_gpt2, workspace_llama160m
            ├── device.json                 # (optional) Device configuration
            ├── kernel_<op>_<rank>.py       # Original kernel implementations
            ├── kernel_<op>_<rank>_optimized.py # Optimized kernel implementations
            ├── optimization_plan.json      # Optimization strategy
            ├── orchestration_state.json    # Execution state
            ├── profile_report.json         # Profiling results
            ├── aggregate_report.md         # (optional) Aggregated results
            ├── results.tsv                 # (optional) Tabular results
            └── results/                    # Output and benchmarks directory
```

### Dimension Mappings

- **`<coding_agent>`**: The AI model used for optimization (e.g., `sonnet`)
- **`<target_device>`**: Target hardware platform (e.g., `h100`, `rtx_5070`)
- **`<model>`**: Model being optimized (e.g., `gpt2`, `llama160m`, `llama7b`)
- **`<op>`**: Operation type (e.g., `gemm`, `flash_attention`, `fused_mlp`, `matmul`, `reduce`)
- **`<rank>`**: Kernel identifier/rank within the model

### Key Files

- **`kernel_<op>_<rank>.py`**: Original kernel implementations
- **`kernel_<op>_<rank>_optimized.py`**: Optimized versions of kernels
- **`optimization_plan.json`**: Strategy for kernel optimization
- **`orchestration_state.json`**: State tracking for optimization process
- **`profile_report.json`**: Performance profiling data
- **`results.tsv`**: Tabular results summary
- **`results/`**: Directory containing detailed output files

