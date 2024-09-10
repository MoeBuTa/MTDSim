import pandas as pd
import os

class SecurityMetricStatistics:
    def __init__(self):
        self._metric_record = {
            'CompleteTopologyShuffle': 0,
            'IPShuffle':0,
            'OSDiversity':0,
            'ServiceDiversity':0
        }

    def increment_metric(self, field_name):
        if field_name in self._metric_record:
            self._metric_record[field_name] += 1
        else:
            print(f"{field_name} is not a valid field in the metric record.")

    def get_record(self):
        df = pd.DataFrame(self._metric_record)
        df = df.drop_duplicates(subset=['times'], keep='last')
        return df
