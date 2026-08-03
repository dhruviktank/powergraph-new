# PowerGraph-Node Reproduction

This package is an independent re-implementation of the PowerGraph-Node benchmark pipeline for node-level PF and OPF regression.

It covers:
- raw `.mat` dataset loading
- graph preprocessing
- max-abs node normalization
- masked regression training
- GAT and Graph Transformer models only
- validation, testing, checkpoint saving
- per-run CSV/JSON output
- multi-run benchmark sweeps and aggregation

## Files

- `data.py`: dataset loading, preprocessing, and processed bundle I/O
- `preprocess.py`: CLI for building the processed train/val/test tensors
- `models.py`: GAT and Graph Transformer implementations
- `train.py`: single-run training entry point and artifact export
- `benchmark.py`: multi-seed / multi-hyperparameter sweep runner
- `aggregate.py`: aggregates saved run summaries into a table

## Usage

### Preprocess

```bash
python -m powergraph_repro.preprocess \
  --data_dir raw_data \
  --out_dir processed_powergraph \
  --problem pf \
  --train_frac 0.85 --val_frac 0.05 --test_frac 0.10 \
  --seed 0
```

### Train one run

```bash
python -m powergraph_repro.train \
  --data_dir raw_data \
  --processed_dir processed_powergraph \
  --results_dir results \
  --model_name transformer \
  --problem pf \
  --hidden_dim 8 --num_layers 1 --heads 4 \
  --epochs 50 --batch_size 32 --lr 1e-3
```

The trainer uses the same regression scheduler as the original benchmark: `ReduceLROnPlateau` on validation loss with `factor=0.1` and `patience=10`. It automatically lowers the learning rate when the reference metric stops improving.

### Run benchmark sweep

```bash
python -m powergraph_repro.benchmark \
  --data_dir raw_data \
  --processed_dir processed_powergraph \
  --results_dir results \
  --model_name transformer \
  --problem nodeopf \
  --seeds 0,100,300,700,1000 \
  --hidden_dims 8,16,32 \
  --num_layers_list 1,2,3
```

### Aggregate saved runs

```bash
python -m powergraph_repro.aggregate \
  --results_dir results \
  --output results/aggregate_summary.csv
```

## Notes

- This implementation keeps the benchmark protocol explicit and reproducible.
- It intentionally uses only GAT and Graph Transformer for model coverage.
- The original paper pipeline saved Excel summaries; this independent implementation saves CSV/JSON artifacts so it works without spreadsheet dependencies.
- Learning-rate decay is driven by validation loss with the original benchmark's plateau settings.
