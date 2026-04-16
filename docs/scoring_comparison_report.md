# ML Architecture Comparison Report: Senior AI vs. Colleague Prototype

## Critical Findings
- **Inverted Polarities:** The Colleague version ranks counties by **descending risk** (Dublin = Rank 1, highest risk). The Senior AI version ranks by **ascending risk** (Kildare/Meath = Rank 1, lowest risk).
- **Metric Scaling:** The Colleague version uses significantly higher base risk scores (Mean: 39.5) compared to the Senior AI formulation (Mean: 27.5), suggesting a more conservative weight bias towards risk in the colleague's model.
- **Anomaly Detection Divergence:** The Colleague's `reconstruction_error` and `confidence` metrics appear to be stochastically distributed or manually weighted, whereas the Senior AI version uses a consistent PCA reconstruction matrix for error calculation.

## Detail Comparison Table
| County | Colleague Rank | Senior Rank | Risk (Colleague) | Risk (Senior) | Risk Delta |
| --- | --- | --- | --- | --- | --- |
| Carlow | 25 | 15 | 20.0 | 29.2 | 9.20 |
| Cavan | 5 | 21 | 45.8 | 32.3 | -13.50 |
| Clare | 19 | 7 | 22.3 | 24.8 | 2.50 |
| Cork | 13 | 9 | 43.2 | 25.5 | -17.70 |
| Donegal | 10 | 25 | 44.2 | 34.8 | -9.40 |
| Dublin | 1 | 16 | 59.9 | 30.0 | -29.90 |
| Galway | 15 | 4 | 39.1 | 24.0 | -15.10 |
| Kerry | 21 | 13 | 21.2 | 28.8 | 7.60 |
| Kildare | 7 | 1 | 45.4 | 14.1 | -31.30 |
| Kilkenny | 20 | 8 | 21.5 | 24.9 | 3.40 |
| Laois | 26 | 14 | 20.0 | 29.2 | 9.20 |
| Leitrim | 14 | 26 | 41.6 | 35.0 | -6.60 |
| Limerick | 16 | 5 | 27.0 | 24.2 | -2.80 |
| Longford | 9 | 24 | 44.8 | 33.5 | -11.30 |
| Louth | 17 | 6 | 26.3 | 24.5 | -1.80 |
| Mayo | 4 | 23 | 45.8 | 32.3 | -13.50 |
| Meath | 12 | 2 | 43.3 | 15.9 | -27.40 |
| Monaghan | 6 | 20 | 45.6 | 32.1 | -13.50 |
| Offaly | 2 | 19 | 47.3 | 30.9 | -16.40 |
| Roscommon | 8 | 22 | 44.9 | 32.3 | -12.60 |
| Sligo | 3 | 18 | 47.0 | 30.7 | -16.30 |
| Tipperary | 23 | 12 | 20.5 | 28.8 | 8.30 |
| Waterford | 18 | 10 | 24.0 | 25.6 | 1.60 |
| Westmeath | 24 | 11 | 20.3 | 26.6 | 6.30 |
| Wexford | 22 | 17 | 20.8 | 30.5 | 9.70 |
| Wicklow | 11 | 3 | 44.1 | 16.5 | -27.60 |

## Recommendation
The Senior AI implementation is technically more robust due to the use of self-supervised GBM targets and actual PCA project error for the `reconstruction_error` metric. However, for UI consistency, we should align the **Ranking Order**. The dashboard currently expects Rank 1 to be the 'best' (lowest risk) area.
