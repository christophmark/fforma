[![Build](https://github.com/FedericoGarza/fforma/workflows/Python%20package/badge.svg?branch=master)](https://github.com/FedericoGarza/fforma/tree/master)
[![PyPI version fury.io](https://badge.fury.io/py/fforma.svg)](https://pypi.python.org/pypi/fforma/)
[![Downloads](https://pepy.tech/badge/fforma)](https://pepy.tech/project/fforma)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/FedericoGarza/fforma/blob/master/LICENSE)

# Installation

```python
pip install git+https://github.com/FedericoGarza/fforma
```

# Usage

See `comparison-fforma-r.ipynb` for an example using the original data.

# Current Results

| DATASET   | OUR OWA | OUR OWA  (W OUR FEATS) | M4 OWA (Hyndman et.al.) |
|-----------|:-------:|:---------------------:|:------------------------:|
| Yearly    | 0.802   | 0.818                 | 0.799  |
| Quarterly | 0.849   | 0.857                 | 0.847  |
| Monthly   | 0.860   | 0.877                 | 0.858  |
| Hourly    | 0.510   | 0.489                 | 0.914  |
| Weekly    | 0.887   | 0.884                 | 0.914  |  
| Daily     | 0.977   | 0.977                 | 0.914  |
