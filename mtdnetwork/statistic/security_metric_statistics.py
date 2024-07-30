import pandas as pd
import os

class SecurityMetricStatistics:
    def __init__(self):
        self._metric_record = []

    def append_security_metric_record(self, state_array, timeseries_array, time):
        self._metric_record.append({
            'host_compromise_ratio': state_array[0],
            'exposed_endpoints': state_array[1],
            'attack_path_exposure_score': state_array[2],
            'overall_asr_ratio': state_array[3],
            'roa': state_array[4],
            'shortest_path_vulnerability': state_array[5],
            'risk': state_array[6],
            'mtd_frequency': timeseries_array[0],
            'overall_mtcc_avg': timeseries_array[1],
            'time_since_last_mtd': timeseries_array[2],
            'times': time
        })
    
    def get_record(self):
        df = pd.DataFrame(self._metric_record)
        df = df.drop_duplicates(subset=['times'], keep='last')
        return df