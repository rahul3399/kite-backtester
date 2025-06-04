# Kite Options Trading - Pair Trading Strategy

A FastAPI-based application for implementing pair trading strategies on Nifty 50 options using Zerodha Kite WebSocket API.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Kite API credentials
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
kite-options-trading/
├── app/                    # Main application package
│   ├── models/            # Pydantic data models
│   ├── services/          # Business logic services
│   ├── api/               # API endpoints
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── logs/                  # Log files (auto-created)
└── requirements.txt       # Python dependencies
```
