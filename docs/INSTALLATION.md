# 🚀 AgentQuant Installation & Setup Guide

## 📋 Prerequisites

### System Requirements
- **Python 3.10+** (3.11 recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 2GB free space for data and dependencies
- **Internet**: Required for data fetching and AI model access

### Required API Keys
1. **Google Gemini API Key** (Required)
   - Visit: https://makersuite.google.com/app/apikey
   - Create account and generate API key
   - Free tier available with generous limits

2. **FRED API Key** (Optional but recommended)
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Free registration required
   - Provides macroeconomic data

## 🛠️ Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# Step 1: Clone the repository
git clone https://github.com/onepunchmonk/AgentQuant.git
cd AgentQuant

# Step 2: Create virtual environment
python -m venv venv

# Step 3: Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Verify installation
python -c "import streamlit; import pandas; import yfinance; print('✅ Installation successful!')"
```

### Method 2: Docker Installation

```bash
# Build the Docker image
docker build -t agentquant .

# Run the container
docker run -p 8501:8501 -v $(pwd)/.env:/app/.env agentquant
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n agentquant python=3.11
conda activate agentquant

# Install pip dependencies
pip install -r requirements.txt
```

## 🔧 Configuration Setup

### Step 1: Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor
notepad .env  # Windows
nano .env     # macOS/Linux
```

Add your API keys:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: FRED API Key for macroeconomic data
FRED_API_KEY=your_fred_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
```

### Step 2: Stock Universe Configuration

Edit `config.yaml` to specify your stock universe:

```yaml
# Example configuration
universe:
  - "AAPL"    # Apple Inc.
  - "MSFT"    # Microsoft Corporation
  - "GOOGL"   # Alphabet Inc.
  - "AMZN"    # Amazon.com Inc.
  - "TSLA"    # Tesla Inc.
  - "NVDA"    # NVIDIA Corporation
  - "SPY"     # SPDR S&P 500 ETF
  - "QQQ"     # Invesco QQQ ETF
  - "TLT"     # iShares 20+ Year Treasury ETF
  - "GLD"     # SPDR Gold Trust ETF

# Additional configuration options
data:
  yfinance_period: "3y"    # Data period: 1y, 2y, 3y, 5y, max
  
backtest:
  initial_cash: 100000     # Starting capital ($100,000)
  commission: 0.001        # Commission rate (0.1%)
  
agent:
  max_strategies: 8        # Number of strategies to generate
```

## 🚀 Running AgentQuant

### Quick Start

```bash
# Ensure you're in the project directory and virtual environment is activated
cd AgentQuant
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Launch the application
python run_app.py
```

The application will start and display:
```
Created figures directory: /path/to/AgentQuant/figures
Starting Streamlit app: /path/to/AgentQuant/src/app/streamlit_app.py

You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

### Alternative Launch Methods

```bash
# Method 1: Direct Streamlit command
streamlit run src/app/streamlit_app.py

# Method 2: Python module execution
python -m streamlit run src/app/streamlit_app.py --server.port 8501

# Method 3: Custom port
python run_app.py --port 8502
```

## 🖥️ Using the Dashboard

### Navigation Overview

1. **Sidebar Controls**
   - Stock selection from your universe
   - Date range picker (start/end dates)
   - Number of strategies to generate
   - Advanced settings

2. **Main Dashboard Tabs**
   - **Strategy Generation**: Create AI-powered strategies
   - **Performance Analysis**: View backtesting results
   - **Strategy Comparison**: Compare multiple strategies
   - **Mathematical Formulas**: View exact strategy equations

### Step-by-Step Usage

#### 1. Select Your Assets
- Use the sidebar to choose stocks from your configured universe
- Typical selections: 3-8 assets for diversified strategies
- Mix asset classes: stocks, ETFs, bonds, commodities

#### 2. Set Analysis Period
- **Start Date**: How far back to analyze (default: 3 years)
- **End Date**: Analysis end point (default: yesterday)
- **Recommendation**: Use at least 2 years for robust results

#### 3. Generate Strategies
- Click **"Generate Strategies"** button
- **Processing time**: 2-5 minutes depending on:
  - Number of assets selected
  - Number of strategies requested
  - Market data to download

#### 4. Review Results
- **Performance Charts**: Interactive equity curves
- **Strategy Details**: Mathematical formulations
- **Risk Metrics**: Sharpe ratio, max drawdown, volatility
- **Allocation Charts**: Dynamic asset weighting

#### 5. Export Results
- **Save Figures**: Automatically saved to `figures/` directory
- **Download Data**: Export performance metrics as CSV
- **Strategy Code**: Copy mathematical formulas

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. Module Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'xyz'
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

#### 2. API Key Issues
```bash
# Error: GOOGLE_API_KEY not found
# Solution: Check .env file exists and contains valid API key
cat .env  # Verify file contents
```

#### 3. Data Download Failures
```bash
# Error: Failed to download data for TICKER
# Solutions:
# - Check internet connection
# - Verify ticker symbols are valid
# - Try with smaller date range
```

#### 4. Performance Issues
```bash
# Issue: Slow processing or memory errors
# Solutions:
# - Reduce number of assets (try 3-5 initially)
# - Shorten date range (try 1-2 years)
# - Close other applications to free memory
```

#### 5. Port Already in Use
```bash
# Error: Port 8501 is already in use
# Solutions:
streamlit run src/app/streamlit_app.py --server.port 8502
# Or kill existing process:
pkill -f streamlit  # macOS/Linux
taskkill /F /IM python.exe  # Windows
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG  # macOS/Linux
set LOG_LEVEL=DEBUG     # Windows

# Run with debug output
python run_app.py 2>&1 | tee debug.log
```

## 📊 Understanding Results

### Performance Metrics Explained

1. **Total Return**: Overall gain/loss percentage
2. **Sharpe Ratio**: Risk-adjusted return measure (>1.0 is good)
3. **Maximum Drawdown**: Largest peak-to-trough decline
4. **Volatility**: Annualized standard deviation of returns
5. **Win Rate**: Percentage of profitable trades

### Strategy Types Generated

- **Momentum**: Trend-following strategies
- **Mean Reversion**: Contrarian strategies
- **Volatility**: Risk-based allocation
- **Multi-Asset**: Cross-asset strategies

### Chart Interpretations

- **Equity Curve**: Portfolio value over time
- **Drawdown Chart**: Portfolio declines from peaks
- **Allocation Chart**: Asset weights over time
- **Rolling Metrics**: Time-varying performance

## 🔄 Data Updates

### Automatic Updates
- Market data refreshes automatically when running
- Historical data cached locally for faster subsequent runs
- Cache location: `data_store/` directory

### Manual Data Refresh
```python
# Force data refresh in Python
from src.data.ingest import fetch_ohlcv_data
data = fetch_ohlcv_data(force_download=True)
```

### Cache Management
```bash
# Clear data cache
rm -rf data_store/  # macOS/Linux
rmdir /s data_store  # Windows

# Clear figure cache
rm -rf figures/
```

## 🔧 Advanced Configuration

### Custom Strategy Parameters

Edit `config.yaml` for advanced customization:

```yaml
# Strategy-specific parameters
strategies:
  momentum:
    fast_window_range: [5, 25]
    slow_window_range: [20, 100]
  
  mean_reversion:
    bollinger_window_range: [10, 50]
    std_dev_range: [1.5, 3.0]
  
  volatility:
    target_vol_range: [0.10, 0.25]
    lookback_range: [20, 100]
```

### Risk Management Settings

```yaml
risk:
  max_position_size: 0.40      # Max 40% in single asset
  max_drawdown: 0.15           # Stop at 15% drawdown
  rebalance_frequency: "monthly"
  stop_loss: 0.05              # 5% stop loss
```

### Performance Optimization

```yaml
performance:
  parallel_processing: true
  max_workers: 4
  batch_size: 100
  cache_enabled: true
```

## 📱 Mobile & Remote Access

### Access from Other Devices

1. **Find your IP address**:
   ```bash
   ipconfig  # Windows
   ifconfig  # macOS/Linux
   ```

2. **Access remotely**: `http://YOUR_IP:8501`

3. **Secure tunnel** (recommended for external access):
   ```bash
   # Using ngrok
   ngrok http 8501
   ```

## 🔒 Security Considerations

### API Key Security
- Never commit `.env` file to version control
- Use environment variables in production
- Rotate API keys regularly

### Data Privacy
- All processing happens locally
- No data sent to external services (except API calls)
- Market data cached locally

### Network Security
- Default setup only accessible locally
- Use VPN for remote access
- Consider firewall rules for production

## 📈 Performance Benchmarks

### Expected Processing Times
- **Small Portfolio** (3 assets, 2 years): 1-2 minutes
- **Medium Portfolio** (5 assets, 3 years): 2-4 minutes  
- **Large Portfolio** (10 assets, 5 years): 5-10 minutes

### Memory Usage
- **Minimum**: 2GB RAM
- **Recommended**: 8GB RAM
- **Large datasets**: 16GB+ RAM

### Storage Requirements
- **Base installation**: 500MB
- **Data cache**: 100-500MB per year of data
- **Results**: 10-50MB per analysis

## 🆘 Getting Help

### Documentation
- **README.md**: Project overview and features
- **DESIGN.md**: Technical architecture details
- **API Documentation**: In-code docstrings

### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share strategies
- **Wiki**: Community-contributed guides

### Professional Support
- **Consulting**: Custom strategy development
- **Training**: Workshop and training sessions
- **Enterprise**: Licensed support options

---

## ✅ Installation Checklist

Use this checklist to verify your installation:

- [ ] Python 3.10+ installed and working
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] `.env` file created with API keys
- [ ] `config.yaml` configured with your stock universe
- [ ] Application starts without errors
- [ ] Dashboard accessible at http://localhost:8501
- [ ] Sample strategy generation completed successfully
- [ ] Results display properly with charts and metrics

**🎉 Congratulations!** You're ready to start using AgentQuant for autonomous trading strategy research!
