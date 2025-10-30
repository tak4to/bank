import pandas as pd
import polars as pl
import seaborn as sns
import sklearn
def library():
    print(f"pandas version:{pd.__version__}")
    print(f"polars version:{pl.__version__}")
    print(f"seaborn bersion:{sns.__version__}")
    print(f"scikit-learn version:{sklearn.__version__}")