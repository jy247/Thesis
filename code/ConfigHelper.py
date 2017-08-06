import json

class SeriesInfo:

    def __init__(self):
        info = []

    info = {'id' : 'DPCERL1Q225SBEA',
            'frequency' : 'Quarterly',
            'units' : 'lin'}

one_si = SeriesInfo()
with open('../config/series_definitions.json', 'w') as outfile:
    json.dump(one_si.info, outfile)

with open('../config/series_definitions.json') as json_data:
    d = json.load(json_data)
    print(d)