import pandas as pd

def get_data(file_name):
    data = []
    raw_data = pd.read_csv(file_name, header=0, names=['id', 'time', 'value', 'label'])
    for name, group in raw_data.groupby('id'):
        group = group.sort_values('time')
        group.name = name
        data.append(group)
    return data
