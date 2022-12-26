import pandas as pd

def test_csv_data(path='postgres/data.csv'):
   assert path.split('.')[-1] == 'csv', 'Data must be .csv'

def test_target(path='postgres/data.csv'):
   data = pd.read_csv(path)
   assert 'target' in data.columns.to_list(), 'Check name of target'

def test_nan_target(path='postgres/data.csv'):
   data = pd.read_csv(path)
   assert data.target.isna().sum() == 0, 'Target has nan values'

def test_target_type(path='postgres/data.csv'):
   data = pd.read_csv(path)
   assert data.target.dtype == 'int64', 'Target must be int64'