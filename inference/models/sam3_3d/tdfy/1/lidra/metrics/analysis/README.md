# Metrics Analysis Tool

A command-line tool for analyzing multi-trial evaluation results in Lidra.

## Overview

This tool processes CSV files containing evaluation metrics from run evaluations; computing statistics across multiple trials per sample and generating formatted reports.

## Functionality

- **Flexible reporting with typed Hydra configs** (`definitions.py`): Configure which metrics to analyze via YAML configs
- **Flexible analysis** (`processor/`): E.g. Analyze experiments with multiple predictions per sample
  - `mean`: Average metrics across all trials
  - `best`: Select best trial per sample based on a key metric
- **Multiple output formats** (`console/`): CSV, TSV, Rich terminal UI 


## Usage

```bash
# Basic usage
python -m lidra.metrics.analysis -i evaluation_results.csv

# Save results to directory with Rich UI output
python -m lidra.metrics.analysis -i results.csv -o output_dir/ --design rich

# Best-of-trials analysis with first 3 trials only
python -m lidra.metrics.analysis -i results.csv --mode best --max-trials 3

# Custom report configuration
python -m lidra.metrics.analysis -i results.csv --report-config custom_report.yaml
```

## Input Format

Expected CSV columns:
- `sample_uuid`: Unique identifier for each sample
- `trial`: Trial number (1-indexed)
- Metric columns: `f1`, `chamfer_distance`, `oriented_rot_error_deg`, etc.



## Output
- **CSV files**: One file per metric table or combined single file
- **Metadata**: Analysis configuration and data summary


## Configuration

Reports are configured via YAML files (see `etc/lidra/reports/tdfy/multi_trial_stage_1.yaml`):
