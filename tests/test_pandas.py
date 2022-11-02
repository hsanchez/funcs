from unittest import TestCase
import pandas as pd
from funcs import pandas as xpd


class TestPandas(TestCase):
  def test_pandas(self):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    self.assertEqual(df.shape, (3, 2))
  
  def test_ordinal_encoding(self):
    df = pd.DataFrame({'rating':['Good', 'Bad', 'Good', 'Good', 
                                 'Bad', 'Neutral', 'Good', 'Good', 
                                 'Neutral', 'Neutral', 'Neutral','Good', 
                                 'Bad', 'Good']})
    df_ord = xpd.ordinal_encode(df, [])
    self.assertEqual(df_ord.shape, (14, 1))
    
  def test_ordinal_encoding_view(self):
    df = pd.DataFrame({'rating':['Good', 'Bad', 'Good', 'Good', 
                                 'Bad', 'Neutral', 'Good', 'Good', 
                                 'Neutral', 'Neutral', 'Neutral','Good', 
                                 'Bad', 'Good']})
    df_ord = xpd.ordinal_encode(df, [], True)
    self.assertEqual(df_ord.shape, (14, 2))

