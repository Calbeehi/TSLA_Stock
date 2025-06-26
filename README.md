# README: Medium-Term and Long-Term Trend Prediction Using LSTM

## Project Overview

This project is designed to forecast financial market trends of Tesla Inc using machine learning models, specifically Long Short-Term Memory (LSTM) networks, for both **medium-term** and **long-term** predictions. The goal is to predict the market trend based on historical data, with three possible outcomes: 

- **Decrease**
- **Stable**
- **Increase**

### Key Features

- **Medium-Term Prediction (H=60)**: Predicts market trends in the next 60 trading days.
- **Long-Term Prediction (H=250)**: Predicts market trends in the next 250 trading days.
- Both models are optimized using **Bayesian Optimization** to fine-tune hyperparameters for better performance.

## Dataset

Several factors can influence stock prices. According to the article *"6 factors affecting stock prices"* from **ASEAN Securities**, here are the key elements:

1. **Historical Price & Volume Data**: 
   - **Price**: Open, High, Low, Close, and Adjusted Close.
   - **Volume**: Trading volume reflects market interest and activity.

2. **Technical Indicators**:
   - **RSI (Relative Strength Index)**: Measures the speed and change of price movements to identify overbought or oversold conditions.
   - **Bollinger Bands**: Measures price volatility and provides insight into potential market reversals.
   - **ATR (Average True Range)**: Reflects market volatility.
  
3. **Market Sentiment & News Data**:
   - **Sentiment Analysis**: Analysis of news articles to determine the market's sentiment (positive, neutral, or negative).
   - **Fear & Greed Index**: Analyzes investor sentiment, such as the VIX (Volatility Index).
   
4. **Macroeconomic Data**:
   - **Interest Rates**: Federal Reserve rates impact stock prices as they influence borrowing costs.
   - **Inflation Rate**: Affects purchasing power and market expectations.
   - **Unemployment Rate**: Reflects economic health, impacting consumer confidence and spending.
   - **Commodity Prices**: E.g., Gold, which acts as a safe-haven asset during times of economic uncertainty.

5. **Time-related Features**:
   - **Day of the Week**: Certain patterns are often observed depending on the day.
   - **Month**: Seasonal trends and market behavior patterns.

## Project Structure

The project is divided into two main parts:
1. **Medium-Term Prediction (H=60)**: Forecasts market trends in the next 60 trading days.
2. **Long-Term Prediction (H=250)**: Forecasts market trends in the next 250 trading days.

Each part includes the following steps:
- **Data Preprocessing**: 
  - Cleans and processes the historical market data.
  - Calculates features like VIX changes, gold price changes, and sentiment scores.
  - Creates lagged features for time-series analysis.
- **Model Building**: 
  - Defines an LSTM-based deep learning model with several hyperparameters (e.g., number of LSTM units, dropout rate, batch size).
  - Uses early stopping and learning rate reduction to improve training.
- **Bayesian Optimization**: 
  - Fine-tunes the model’s hyperparameters to achieve optimal performance.
- **Model Evaluation**: 
  - Uses accuracy, precision, recall, and F1 score for evaluation.

## Requirements

The following libraries are required to run this project:

- **Python**: Version 3.6+
- **Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - matplotlib
  - seaborn
  - scikit-optimize

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn scikit-optimize
```
## Results
- Medium-Term Prediction (H=60): Accuracy: 87.25%
- Long-Term Prediction (H=250): Accuracy: 81.79%