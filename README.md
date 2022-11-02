# SIGNAL Utils

## Description 👇
Collection of helper functions for SIGNAL in Google Colab Notebooks

## How to run

Within Google Colabs
```shell
# install
!pip install git+https://github.com/hsanchez/funcs.git
```

Import signal_utils's colab helper functions
```shell
#%%
from funcs import colabs as cl
from funcs.console import stdout

#%%
stdout.print(cl.is_run_in_colab())
# True
```

Create an interactive plot using plotly
```shell
##%
import plotly
import pandas as pd
import matplotlib.pyplot as plt

from funcs.plots as pl

#%%
pl.init_plotly_notebook_mode()

#%%
df = pd.DataFrame({
    'month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Year_2018': [3.26, 6.11, 4.86, 6.53, 4.45, 3.86, 8.04, 7.59, 1.48, 4.75, 7.27, 11.83],
    'Year_1996': [8.26, 3.82, 6.42, 2.91, 2.12, 1.70, 2.14, 4.66, 4.32, 0.89, 3.22, 4.14]})
df.head(1)
# 	month	Year_2018	Year_1996
# 0	January	3.26	8.26

#%%
pl.configure_plotly_browser_state(display_fn=display)
df.iplot(
  kind='line',
  x='month',
  y=['Year_2018', 'Year_1996'],
  color=['white', 'gold'],
  theme='solar',
  mode='markers+lines',
  title='Annual Rainfall in the city Peachtree City, GA')
plt.show()
```

And even describe your dataframes
```shell
#%%
from funcs import pandas as pds
from funcs.console import stdout

#%%
pds.describe_dataframe(df)
# ....
```

## What's inside

In summary, `funcs` has a collection of helper functions
covering numpy, pandas, colabs, Google drive, and plots.

```shell
#%%
from helpfunc import colabs, console, gdrive, numpy, pandas, plots
```


### 🤝 Contributions

We have a [set of guidelines](CONTRIBUTING.md) for contributing to `funcs`.  These are guidelines, not rules. Use your best judgment, and feel free
to propose  changes to this document in a pull request.
