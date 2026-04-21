import numpy as np
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from typing import List


def parse_time_strings(time_list: List[str]) -> List[float]:
    #TODO
    timestamps=[]
    for t in time_list:
        dt = datetime.strptime(t,"%Y-%m-%d %H:%M:%S")
        timestamps.append(dt.timestamp())
    return timestamps

def compute_average_time_diff(timestamps: List[float]) -> float:
    #TODO
    if len(timestamps)<2:
        return 0.0

    sorted_ts = sorted(timestamps)
    diffs = []

    for i in range(len(sorted_ts)):
        for j in range(i + 1, len(sorted_ts)):
            diffs.append(sorted_ts[j] - sorted_ts[i])

    return float(np.mean(diffs))


def hierarchical_clustering(timestamps: List[float], distance_threshold: float) -> List[int]:
    #TODO
    X=np.array(timestamps).reshape(-1,1)
    model =AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average'
    )
    labels=model.fit_predict(X)
    return labels.tolist()

if __name__ == "__main__":
    time_strings = [
        "2023-08-01 12:00:00",
        "2023-08-01 12:05:00",
        "2023-08-01 13:00:00",
        "2023-08-01 15:00:00",
        "2023-08-03 00:00:00",
        "2023-08-05 10:03:01",
        "2023-08-03 09:44:10",
        "2023-08-03 00:00:00",
    ]
    
    timestamps = parse_time_strings(time_strings)
    avg_diff = compute_average_time_diff(timestamps)
    labels = hierarchical_clustering(timestamps, avg_diff)
    
    print("Timestamps:", timestamps)
    print("Average Time Difference:", avg_diff)
    print("Cluster Labels:", labels)