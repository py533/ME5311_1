# ME5311 Project 1 Analysis

A compact Python workflow for POD-based flow analysis, spectral analysis, and figure/table export.

## Project Files
- `main.py`: full analysis pipeline entry point
- `config.py`: paths and analysis parameters
- `load_data.py`: data loading and frame indexing
- `analysis.py`: POD, spectra, PSD, and metrics functions
- `plot.py`: consolidated publication-style figure generation
- `data/vector_64.npy`: input dataset (not included — see `data/placeholder.txt`)

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
From the project root:

```bash
python main.py
```

## Outputs
- Figure: `outputs/q1_q4_summary_B.png`
- Text summary: `outputs/analysis_summary.txt`
