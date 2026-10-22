# Time Series Forecasting Using Ghana's 91-Day Treasury Bill Rate

## Overview

This project investigates the historical behaviour and forecasting of Ghana's 91-day Treasury bill rates using traditional time-series models and machine-learning approaches.

Completed as my final-year research project for my Statistics degree at Accra Technical University, the study explores how different modelling approaches can be applied to financial time-series data.

## Research Objectives

- Examine historical patterns and variability in Ghana's 91-day Treasury bill rates.
- Investigate stationarity and relevant time-series characteristics.
- Compare traditional forecasting methods with machine-learning approaches.
- Examine model diagnostics and forecasting behaviour.
- Consider the practical limitations of forecasting financial rates.

## Models Examined

- ARIMA
- SARIMA
- Holt-Winters exponential smoothing
- Prophet
- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)

## Data and Methodology

The thesis reports 381 observations covering June 2017 to May 2025. The analysis includes descriptive statistics, stationarity testing, time-series modelling, and model comparisons.

The original thesis contains some inconsistencies in its stated study-period dates. These should be reconciled against the underlying dataset and analysis before presenting this repository as a fully reproducible study.

## Key Findings Reported in the Thesis

- The series exhibited non-stationary behaviour before differencing.
- ARIMA(1,1,0) had the lowest reported Akaike Information Criterion (AIC) among the discussed traditional models, with an AIC of 761.95.
- The thesis describes changes in the Treasury bill rate's behaviour during 2022–2023.
- The reported ANN and LSTM results showed different generalisation and forecasting behaviours.

**Interpretation note:** A lower AIC supports model selection under the criterion's assumptions; it does not, by itself, establish superior out-of-sample forecasting accuracy. Comparative accuracy claims should be supported by verified test-set metrics and an appropriate evaluation procedure.

## Research Scope

This project focuses on statistical modelling, financial time series, forecasting, and the comparison of econometric and machine-learning methods in the context of Ghana's fixed-income market.

## Limitations and Future Work

Potential extensions include:

- Use a clearly documented chronological train/validation/test split.
- Compare models using appropriate out-of-sample error metrics.
- Investigate rolling-window or walk-forward validation.
- Examine the effects of structural breaks and changing market regimes.
- Evaluate forecast uncertainty and prediction intervals.
- Document the data source, preprocessing, dependencies, and random seeds for reproducibility.

## Project Report

The final-year thesis will be included in the `report/` directory, subject to any applicable university requirements and permissions.

## Author

David Quayefio

- LinkedIn: https://www.linkedin.com/in/david-quayefio/
- GitHub: https://github.com/davidquayefio

## Disclaimer

This is an academic research project, not investment advice. Historical findings and forecasts should not be interpreted as current market predictions.
