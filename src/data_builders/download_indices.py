import yfinance as yf
import polars as pl
import pathlib

def download_and_save():
    print("Downloading S&P 500 data...")
    # Download S&P 500
    sp500_df = yf.download('^GSPC', start='2005-01-01')
    
    # yf.download returns a pandas DataFrame with MultiIndex columns in recent versions
    # We will reset index to get Date, and then convert to Polars
    sp500_pandas = sp500_df.reset_index()
    # Flatten multi-index columns if they exist
    if isinstance(sp500_pandas.columns, filter):
        pass # Handle multi-index below if needed, though simple download usually single
    if hasattr(sp500_pandas.columns, "levels"):
        sp500_pandas.columns = ['_'.join(col).strip('_') for col in sp500_pandas.columns.values]
    
    sp500_pl = pl.from_pandas(sp500_pandas)
    
    print("Downloading DXY data...")
    # Download DXY
    dxy_df = yf.download('DX-Y.NYB', start='2005-01-01')
    dxy_pandas = dxy_df.reset_index()
    
    if hasattr(dxy_pandas.columns, "levels"):
        dxy_pandas.columns = ['_'.join(col).strip('_') for col in dxy_pandas.columns.values]
    
    dxy_pl = pl.from_pandas(dxy_pandas)
    
    raw_data_dir = pathlib.Path("raw_data")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    sp500_path = raw_data_dir / "sp500_daily.parquet"
    dxy_path = raw_data_dir / "dxy_daily.parquet"
    
    sp500_pl.write_parquet(sp500_path)
    print(f"Saved S&P 500 to {sp500_path}")
    
    dxy_pl.write_parquet(dxy_path)
    print(f"Saved DXY to {dxy_path}")

if __name__ == "__main__":
    download_and_save()
