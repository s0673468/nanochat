import pandas as pd

# Creating a sample dataframe
df = pd.DataFrame({'A': range(100000), 'B': ['data'] * 100000})

# Save as Parquet (usually much smaller than the equivalent CSV)
df.to_parquet('example.parquet', engine='pyarrow', compression='snappy')

# Read it back
df_loaded = pd.read_parquet('example.parquet')