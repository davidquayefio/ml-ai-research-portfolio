# Forecasting Ghana’s 91-Day Treasury Bill Rates

## Overview

This project presents my final-year research in Statistics at Accra Technical University. It investigates time-series forecasting methods for Ghana’s 91-day Treasury bill rate, comparing traditional statistical models with machine learning and deep learning approaches.

The research explores how different modelling approaches capture changes in Treasury bill rates and examines the challenges of forecasting a financial time series characterised by periods of substantial volatility.

## Research Objectives

The study aims to:

- Examine the historical behaviour and statistical properties of Ghana’s 91-day Treasury bill rate.
- Investigate stationarity and apply differencing where appropriate.
- Compare statistical, machine learning, and deep learning forecasting approaches.
- Examine reported model results and their implications for financial time-series forecasting.
- Identify limitations and opportunities for further research.

## Models Examined

The thesis investigates six forecasting approaches:

| Model | Approach |
|---|---|
| ARIMA | Autoregressive Integrated Moving Average |
| SARIMA | Seasonal Autoregressive Integrated Moving Average |
| Holt-Winters | Exponential smoothing |
| Prophet | Time-series forecasting framework |
| ANN | Artificial Neural Network |
| LSTM | Long Short-Term Memory neural network |

These models represent different approaches to capturing temporal patterns, trends, and nonlinear relationships in time-series data.

## Data and Methodology

The thesis reports **381 daily observations** of Ghana’s 91-day Treasury bill rate, covering June 2017 to May 2025.

The reported methodology includes:

- Exploratory analysis of the rate series.
- Assessment of stationarity using the Augmented Dickey-Fuller (ADF) test.
- First-order differencing to address nonstationarity.
- Statistical model specification and comparison.
- Evaluation of traditional forecasting methods alongside neural-network approaches.

**Data-period note:** The thesis contains inconsistent study-period references across sections. The observation count and date range above reflect the reported data description and should be reconciled against the original dataset before the analysis is reproduced.

## Reported Findings

The thesis reports several findings from its analysis:

- The Treasury bill rate exhibited substantial variation over the period studied, including a period of elevated rates and volatility around 2022–2023.
- The original series was reported as nonstationary, motivating first-order differencing.
- The thesis reports an AIC of 761.95 for ARIMA(1,1,0) and 793.84 for SARIMA(1,1,1)(1,1,0,52).
- The ANN was described as demonstrating strong generalisation, while the LSTM was reported to show overprediction and lagging behaviour on the test data.

These are findings as reported in the thesis, not independently reproduced results. AIC values alone do not establish which model produces the most accurate out-of-sample forecasts. A rigorous comparison would also require clearly documented test periods, forecast errors, evaluation procedures, and reproducible experiments.

## Limitations

The research should be interpreted within the following limitations:

- Inconsistent date references across the thesis need to be reconciled.
- The original dataset and complete analysis code are not currently documented here as a reproducible research package.
- Model performance claims require verification against the underlying predictions and evaluation metrics.
- The reported analysis concerns historical observations and should not be interpreted as a current Treasury bill rate forecast or investment recommendation.

## Future Research

Potential extensions include:

1. Reconcile the study period and validate the underlying dataset.
2. Publish the data-processing and modelling code where permitted.
3. Evaluate models using chronological train-validation-test splits and rolling-origin backtesting.
4. Compare forecast errors using appropriate metrics such as MAE and RMSE.
5. Investigate parameter sensitivity, forecast stability, and performance across different market regimes.
6. Assess whether additional economic or financial variables improve forecasting performance.

## Project Report

Read the complete final-year thesis:

[**View or download the final-year thesis (PDF)**](Final%20Year%20Thesis.pdf)

## Author

**David Quayefio**

BTech Statistics — Accra Technical University

- [LinkedIn](https://www.linkedin.com/in/david-quayefio/)
- [GitHub](https://github.com/davidquayefio)

## Disclaimer

This repository documents academic research on historical Treasury bill rates. The findings are presented for educational and research purposes only. They do not constitute investment advice, and no claim of current forecasting accuracy or investment profitability is made.
