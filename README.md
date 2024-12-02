# Risk-vs-Return-Cluster-Model

# Risk-vs-Return-Cluster-Model

This project aims to analyze and cluster S&P 500 companies based on their risk (Beta) and return (1-Year Return %). The clustering is performed using the K-Means algorithm, and the results are visualized in a scatter plot.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
  - [dataset_script.py](#dataset_scriptpy)
  - [model.py](#modelpy)
- [License](#license)

## Requirements

- yfinance
- pandas
- beautifulsoup4
- requests
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Risk-vs-Return-Cluster-Model.git
    cd Risk-vs-Return-Cluster-Model
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Fetch the latest S&P 500 stock data and save it to a CSV file (current csv could be outdated):
    ```sh
    python dataset_script.py
    ```

2. Run the clustering analysis and generate the visualization:
    ```sh
    python model.py
    ```

## File Structure
```
Risk-vs-Return-Cluster-Model/
├── data/
│   └── sp500_data.csv
├── scripts/
│   ├── dataset_script.py
│   └── model.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Scripts

### dataset_script.py

This script fetches the latest stock data for S&P 500 companies from Yahoo Finance and saves it to a CSV file.

- **Functions:**
  - `get_sp500_tickers()`: Retrieves the list of S&P 500 tickers from Wikipedia.
  - `fetch_stock_data()`: Fetches stock data for each ticker and saves it to a CSV file.

### model.py

This script performs data preprocessing, feature scaling, K-Means clustering, and visualization of the clusters.

- **Functions:**
  - `preprocess_data(df)`: Preprocesses the data by removing missing values, converting percentages to decimals, and excluding outliers.
  - `scale_features(data)`: Scales the features using `StandardScaler`.
  - `perform_kmeans(data)`: Performs K-Means clustering on the scaled data.
  - `analyze_clusters(data, original_data, clusters)`: Analyzes the clusters and calculates statistics for each cluster.
  - `plot_clusters(data, original_data, clusters, scaler, output_filename='clusters.png')`: Plots the clusters and saves the visualization to a file.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.