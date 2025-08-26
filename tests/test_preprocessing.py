import pandas as pd
from scripts.data_preprocessing import clean

def test_clean_shapes():
    import pandas as pd
    df = pd.DataFrame({'a':[1,2,None], 'b':[None,2,3]})
    out = clean(df)
    assert out.isna().sum().sum()==0
