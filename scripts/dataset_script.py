#Script used to create the dataset

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import datetime

def get_sp500_tickers():
    #Get S&P 500 tickers from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    return [row.findAll('td')[0].text.strip() for row in table.findAll('tr')[1:]]

def fetch_stock_data():
    #Fetch stock data and save to CSV
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 companies")

    data = []
    for i, ticker in enumerate(tickers, 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            # Calculate 1-year return
            if not hist.empty:
                first_price = hist['Close'].iloc[0]
                last_price = hist['Close'].iloc[-1]
                year_return = ((last_price - first_price) / first_price) * 100
            else:
                year_return = None

            stock_data = {
                'Symbol': ticker,
                'Name': info.get('longName', None),
                'Sector': info.get('sector', None),
                'Beta': info.get('beta', None),
                'Market Cap': info.get('marketCap', None),
                'Current Price': info.get('currentPrice', None),
                'P/E Ratio': info.get('trailingPE', None),
                'Dividend Yield': info.get('dividendYield', None),
                '1Y Return %': round(year_return, 2) if year_return is not None else None
            }

            data.append(stock_data)
            print(f"Processed {i}/{len(tickers)}: {ticker}")

            # Add small delay to prevent rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"Error with {ticker}: {str(e)}")
            continue

    # Create and save DataFrame
    df = pd.DataFrame(data)

    # Format dividend yield as percentage
    df['Dividend Yield'] = df['Dividend Yield'].apply(
        lambda x: round(x * 100, 2) if pd.notnull(x) else None
    )

    # Format market cap in billions
    df['Market Cap'] = df['Market Cap'].apply(
        lambda x: round(x / 1e9, 2) if pd.notnull(x) else None
    )

    # Rename columns for clarity
    df = df.rename(columns={
        'Market Cap': 'Market Cap (B)',
        'Dividend Yield': 'Dividend Yield %'
    })

    # Save to CSV
    filename = f'sp500_data_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")

    return df

if __name__ == "__main__":
    fetch_stock_data()