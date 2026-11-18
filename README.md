# OracleAgent

A stock-research prototype that combines market bars with structured news sentiment. The FastAPI demo presents market data and a short-horizon estimate. Separate Python scripts collect historical features and train experimental LightGBM models.

| Path | Purpose |
|---|---|
| `api_server.py`, `index.html` | API and browser demo |
| `data_harvester.py`, `sentiment_agent.py` | News retrieval and Gemini feature extraction |
| `train.py`, `train_real.py` | Experimental training scripts |
| `web.py` | Streamlit experiment |

## Run locally

Create and activate a Python virtual environment. Set `GOOGLE_API_KEY` for Gemini and `FINNHUB_API_KEY` for the Finnhub integration. Use your own credentials.

```sh
python -m pip install -r requirements.txt
python -m uvicorn api_server:app --host 127.0.0.1
```

Open [the local demo](http://127.0.0.1:8000/). Live market and news sources require network access. Estimates are experimental and have not been independently backtested.
