# ⚡ QuickSort Performance Analyzer
**AI-powered parallel vs sequential sorting benchmarker with ML predictions and live insights.**

---

## 📁 Project Structure

```
quicksort_analyzer/
├── app.py           # Main Streamlit dashboard
├── sequential.py    # Sequential quicksort + dataset generation
├── parallel.py      # MPI parallel quicksort worker
├── analytics.py     # Metrics engine + insight generator + persistence
├── ml_model.py      # Regression prediction model (numpy-based)
├── utils.py         # MPI runner, simulation fallback, CPU info
└── requirements.txt
```

---

## 🚀 Setup & Run

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install MPI (for real parallel execution)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install open-mpi
```

**Windows:**
Download and install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
then install `mpi4py` via pip.

> **Note:** If MPI is not installed, the app falls back to Amdahl's Law simulation automatically — all features still work.

### 3. Launch the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🎯 Features

| Feature | Description |
|---|---|
| Sequential Quicksort | Pure Python middle-pivot quicksort |
| Parallel Quicksort | MPI scatter → local sort → heap merge |
| Dataset Types | Random, Sorted, Reverse, Nearly Sorted |
| Metrics | Time, Speedup, Efficiency, Throughput |
| AI Insights | Automated text analysis of results |
| ML Prediction | Numpy polynomial regression |
| Auto Benchmark | Sweeps all sizes × process counts |
| Smart Recommender | Suggests optimal config from history |
| Charts | Plotly interactive: time, speedup, efficiency heatmap, scaling |
| Export | CSV + JSON download |
| MPI Fallback | Amdahl's Law simulation if MPI unavailable |

---

## 💡 Usage Tips

1. **Start with Auto Benchmark** to populate data across many configurations.
2. **Train ML Model** after 4+ runs for execution time predictions.
3. **Switch dataset types** to see how input distribution affects performance.
4. **Use 2–4 processes** for best efficiency on local machines.
5. **Check the Efficiency Heatmap** to find the optimal n × processes sweet spot.

---

## 📊 Sample Results

| n | Processes | Seq Time | Par Time | Speedup | Efficiency |
|---|---|---|---|---|---|
| 1,000 | 2 | 0.002s | 0.018s | 0.11× | 5.5% |
| 10,000 | 2 | 0.024s | 0.031s | 0.77× | 38.7% |
| 50,000 | 4 | 0.180s | 0.098s | 1.84× | 45.9% |

> Small inputs: sequential wins. Large inputs: parallel pays off.

---

## ⚙️ Architecture Notes

- `parallel.py` uses MPI scatter/gather with **heap merge** for correct sorted output.
- Timing uses `MPI.Wtime()` for process-synchronized measurement.
- ML model uses 6-feature polynomial regression (n, log n, n·log n, p, 1/p, n/p).
- Results persist in `benchmark_results.json` across sessions.
