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
# if local
stdout.print(cl.is_run_in_colab())
# True
# in Colabs
from IPython import get_ipython
stdout.print(cl.is_run_in_colab(__builtins__, get_ipython()))
```

## What's inside

In summary, `funcs` has a collection of helper functions
covering numpy, pandas, colabs, Google drive, and plots.

```shell
#%%
from helpfunc import colabs, console, gdrive, arrays, data, nlp, plotz
```

### 🤝 Contributions

We have a [set of guidelines](CONTRIBUTING.md) for contributing to `funcs`.  These are guidelines, not rules. Use your best judgment, and feel free
to propose  changes to this document in a pull request.

### ⚖️ License

This project is under the MIT License. See the [LICENSE](https://github.com/hsanchez/funcs/blob/main/LICENSE) file for the full license text.
