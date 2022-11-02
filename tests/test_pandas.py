from unittest import TestCase
import pandas as pd
from funcs import pandas as xpd
from funcs.console import stdout

TEST_DATA = ['Good', 'Bad', 'Good', 'Good', 'Bad', 'Neutral', 'Good', 'Good', 
             'Neutral', 'Neutral', 'Neutral','Good', 'Bad', 'Good']
TEST_DF = pd.DataFrame({'rating': TEST_DATA})

class TestPandas(TestCase):  
  def test_pandas(self):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    self.assertEqual(df.shape, (3, 2))
  
  def test_ordinal_encoding(self):
    df_ord = xpd.ordinal_encode(TEST_DF.copy(), [])
    self.assertEqual(df_ord.shape, (14, 1))
    
  def test_ordinal_encoding_view(self):
    df_ord = xpd.ordinal_encode(TEST_DF.copy(), [], True)
    self.assertEqual(df_ord.shape, (14, 2))
    
  def test_description(self):
    input_df = TEST_DF.copy()
    self.assertTrue(input_df.shape[0] > 0 and input_df.shape[1] > 0)
    input_df_corr = xpd.find_correlated_pairs(input_df)
    self.assertEqual(len(input_df_corr), 0)


