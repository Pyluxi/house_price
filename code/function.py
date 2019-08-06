def write_info(data,name):
    #data:train set or test_set,name:txt name
    import io
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    with open('../tmp/{}_info.txt'.format(name), 'w', encoding='utf-8') as f:
        f.write(s)

def missing_count(data):
    import pandas as pd
    na_count = data.isnull().sum().sort_values(ascending=False)
    na_ratio = na_count / len(data)
    na_data = pd.concat([na_count, na_ratio], axis=1, keys=['count', 'ratio'])
    return na_data