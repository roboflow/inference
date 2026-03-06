# Lidra Metrics Analysis v3

## Overview

The v3 metrics analysis system provides a modular, extensible framework for analyzing experimental metrics data. It uses a template method pattern for processing pipelines, statistical functionals for computations, and Hydra for configuration management.

## Architecture

### Core Components

```
lidra/metrics/analysis/v3/
├── processor/
│   ├── base.py              # Abstract base processor with template method pattern
│   ├── multi_trial.py       # Multi-trial experiment processor
│   ├── formatting.py        # Rich terminal output formatting
│   └── logging.py           # Structured JSON logging
├── functionals.py           # Statistical operations (mean, std, quantiles, etc.)
├── trial_reduction.py       # Strategies for reducing multiple trials
└── cli.py                   # Hydra-based command-line interface
```

### Processing Pipeline

The base `Processor` class defines a standard pipeline:

1. **load_data()** → Load raw data from source (returns DataFrame)
2. **extract_data()** → Extract relevant fields and metadata
3. **transform_data()** → Apply transformations and filtering
4. **compute_results()** → Perform statistical computations (returns DataFrame)
5. **handle_output()** → Format and save results

### Type Safety

The v3 system uses pandas DataFrames for type-safe data handling:
- Input data is loaded as DataFrames
- Statistical results are returned as DataFrames with metrics as index and statistics as columns
- This provides better type safety and easier data manipulation compared to nested dictionaries

## Processors

### MultiTrialProcessor

Analyzes experiments with multiple trials per sample, computing statistics across trials and samples.

#### Configuration

```python
@dataclass
class MultiTrialConfig(BaseConfig):
    # Required inputs
    input_file: str                    # Path to metrics CSV file
    tables: List[str]                  # Metric tables to analyze
    
    # Metrics definition
    metrics: Dict[str, Metric]         # Metric configurations
    thresholds: Dict[str, List[float]] # Threshold values for metrics
    
    # Trial reduction
    trial_reduction_mode: str = "mean" # How to aggregate trials: "mean", "best_by_group", "best_independent"
    
    # Statistics to compute
    statistics: List[str] = ["mean", "std", "p5", "p95"]
    
    # Data cleaning
    drop_na: bool = False              # Drop rows with missing values
    drop_na_columns: List[str] = None  # Drop rows missing these columns
    drop_incomplete_trials: bool = True # Drop samples with incomplete trials
    max_trials: int = None             # Keep only first N trials
    
    # Output options
    save_dir: str = None               # Directory to save results
    console_format: str = "rich"       # Console output format
```

#### Trial Reduction Modes

- **`mean`**: Average all trials per sample
- **`best_by_group`**: Select best trial based on metric groups
- **`best_independent`**: Select best trial independently for each metric

#### Output Format

The processor generates hierarchical metric displays:

```
Table: shape
┌─────────────┬────────┬────────┬────────┬────────┐
│ Metric      │ mean   │ std    │ p5     │ p95    │
├─────────────┼────────┼────────┼────────┼────────┤
│ chamfer_l1 ↓│ 0.0234 │ 0.0089 │ 0.0123 │ 0.0456 │
│ _th_0.01    │ 0.234  │ 0.089  │ 0.123  │ 0.456  │
│ _th_0.02    │ 0.567  │ 0.123  │ 0.234  │ 0.789  │
│             │        │        │        │        │
│ iou ↑       │ 0.6789 │ 0.0234 │ 0.6234 │ 0.7234 │
│ _th_0.5     │ 0.823  │ 0.045  │ 0.756  │ 0.912  │
└─────────────┴────────┴────────┴────────┴────────┘
```

## Statistical Functionals

Available functionals for computing statistics:

### Basic Statistics
- `mean`: Arithmetic mean
- `std`: Standard deviation
- `median`: Median value
- `min`, `max`: Minimum and maximum values

### Quantiles
- `p5`, `p25`, `p50`, `p75`, `p95`: Percentiles
- `quantile`: Custom percentile (requires `q` parameter)
- `iqr`: Interquartile range (p75 - p25)

### Advanced Statistics
- `bootstrap_ci`: Bootstrap confidence intervals
  - Parameters: `n_bootstrap=1000`, `ci_level=0.95`, `agg_func='mean'`

### Creating Custom Functionals

```python
from lidra.metrics.analysis.v3.functionals import StatisticalFunctional, register_functional

@register_functional("custom_stat")
class CustomStatistic(StatisticalFunctional):
    def compute(self, data: np.ndarray) -> float:
        # Your computation here
        return result
```

## Usage

### Basic Console Output

```bash
# Analyze metrics with default settings
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv"

# Specify tables to analyze
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv" \
    "+processor.tables=[shape,oriented]"
```

### Save Results to Directory

```bash
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv" \
    "+processor.save_dir=./results"
```

### Advanced Configuration

```bash
# Use best trial selection
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv" \
    processor.trial_reduction_mode=best_by_group

# Custom statistics
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv" \
    "+processor.statistics=[mean,median,p25,p75,bootstrap_ci]"

# Data cleaning
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv" \
    processor.drop_na=true \
    processor.max_trials=3
```

### Using Configuration Files

Create a custom configuration in `etc/lidra/run/analysis/multi_trial/`:

```yaml
# custom_analysis.yaml
defaults:
  - stage1  # Inherit from base config

processor:
  statistics: [mean, std, median, iqr]
  trial_reduction_mode: best_independent
  drop_incomplete_trials: true
```

Then use it:

```bash
python -m lidra.metrics.analysis.v3.cli \
    --config-name custom_analysis \
    "+processor.input_file=/path/to/metrics.csv"
```

## Extending the System

### Creating a New Processor

1. Create a new processor class inheriting from `Processor`:

```python
from dataclasses import dataclass
import pandas as pd
from .base import Processor, BaseConfig

@dataclass
class MyProcessorConfig(BaseConfig):
    # Your configuration fields
    my_param: str = "default"

class MyProcessor(Processor):
    def load_data(self) -> pd.DataFrame:
        # Load your data
        pass
    
    def extract_data(self, raw_data: pd.DataFrame) -> Any:
        # Extract relevant data
        pass
    
    def transform_data(self, data: Any) -> Any:
        # Apply transformations
        pass
    
    def compute_results(self, data: Any) -> pd.DataFrame:
        # Compute statistics and return as DataFrame
        pass
    
    def handle_output(self, results: pd.DataFrame) -> None:
        # Format and save output
        pass
```

2. Create a Hydra configuration for your processor
3. Update the CLI to support your processor

### Adding Output Formats

The system uses a modular formatting approach. To add a new format:

1. Create a formatter class in `processor/formatting.py`
2. Implement the formatting logic
3. Register it in the processor's `handle_output()` method

## Configuration Reference

### Hydra Override Syntax

- Add new parameter: `"+key=value"`
- Modify existing: `"key=value"`
- Nested parameters: `"processor.key=value"`
- Lists: `"+processor.tables=[table1,table2]"`
- Dictionaries: `"+processor.metrics={metric1:{direction:minimize}}"`

### Environment Variables

- `HYDRA_FULL_ERROR=1`: Show full error traces
- `LIDRA_SKIP_INIT=1`: Skip heavy imports during CLI startup

## Data Format

The system expects CSV files with the following structure:

```csv
sample_uuid,trial,metric1,metric2,...
uuid1,0,0.123,0.456,...
uuid1,1,0.124,0.457,...
uuid2,0,0.234,0.567,...
```

Required columns:
- `sample_uuid`: Unique identifier for each sample
- `trial`: Trial index (0-based)
- Metric columns: Numeric values for each metric

## Logging and Debugging

The system provides structured JSON logging during data extraction:

```bash
# Enable verbose logging
python -m lidra.metrics.analysis.v3.cli \
    "+processor.input_file=/path/to/metrics.csv" \
    processor.verbose=true
```

Logs include:
- Data loading progress
- Cleaning operations (rows dropped, reasons)
- Computation details
- Performance metrics