import urllib.request
import csv
import numpy as np
import torch
import math
from sklearn.preprocessing import StandardScaler

from config import Config, feature_extraction, url_no_label, url_label

vehicle = "2016-chevrolet-silverado"
type = "extra-attack-free"
def download_can_train_and_test_dataset(vehicle, dtype, labeled):
    assert(0), "automatic download currently broken. deciding if it is necessary"
    # if (labeled):
    #     url = url_label + vehicle + "/post-attack-labeled/" + dtype + "-attacks/" + dtype + "-1.csv"
    # else:
    #     url = 
    # url =  \
    #     + vehicle + "/" + type + "/" + type + "-1.csv"
    # output_filename = "data\\" + vehicle + "-" + type + ".csv"

    # try:
    #     urllib.request.urlretrieve(url, output_filename)
    #     print(f"File downloaded successfully to {output_filename}")
    # except Exception as e:
    #     print(f"Error during download: {e}")
def get_stats(vehicle, dtype, labeled):
    """
    Get statistics about a CAN bus dataset.
    
    Args:
        vehicle: Vehicle identifier string
        dtype: Dataset type (e.g., 'extra-attack-free', 'fuzzing')
        labeled: Boolean indicating if dataset has attack labels
    
    Returns:
        Dictionary containing dataset statistics
    """
    raw_data = load_data(vehicle, dtype, labeled)
    
    if isinstance(raw_data, int) and raw_data == -1:
        return {"error": "Dataset not found"}
    
    stats = {}
    
    # Basic counts
    stats['total_messages'] = len(raw_data) - 1  # Subtract header row
    
    if labeled:
        # Attack vs normal message counts
        labels = raw_data[1:, 3].astype(int)
        stats['attack_messages'] = np.sum(labels == 1)
        stats['normal_messages'] = np.sum(labels == 0)
        stats['attack_percentage'] = (stats['attack_messages'] / stats['total_messages']) * 100
        stats['class_imbalance_ratio'] = stats['normal_messages'] / max(stats['attack_messages'], 1)
    else:
        stats['attack_messages'] = 0
        stats['normal_messages'] = stats['total_messages']
        stats['attack_percentage'] = 0.0
        stats['class_imbalance_ratio'] = None
    
    # Temporal statistics
    timestamps = raw_data[1:, 0].astype(float)
    stats['duration_seconds'] = timestamps[-1] - timestamps[0]
    stats['messages_per_second'] = stats['total_messages'] / stats['duration_seconds']
    
    # CAN ID statistics
    can_ids = raw_data[1:, 1]
    unique_ids = np.unique(can_ids)
    stats['unique_can_ids'] = len(unique_ids)
    stats['most_common_id'] = None
    stats['most_common_id_count'] = 0
    
    # Find most common CAN ID
    id_counts = {}
    for can_id in can_ids:
        id_counts[can_id] = id_counts.get(can_id, 0) + 1
    if id_counts:
        most_common = max(id_counts.items(), key=lambda x: x[1])
        stats['most_common_id'] = most_common[0]
        stats['most_common_id_count'] = most_common[1]
        stats['most_common_id_percentage'] = (most_common[1] / stats['total_messages']) * 100
    
    # Data field statistics (non-empty data fields)
    data_fields = raw_data[1:, 2]
    non_empty_data = np.sum([1 for d in data_fields if d and d != '0'])
    stats['messages_with_data'] = non_empty_data
    stats['messages_without_data'] = stats['total_messages'] - non_empty_data
    
    return stats


def print_stats(stats):
    """Pretty print statistics dictionary"""
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return
    
    print("=" * 60)
    print("CAN Bus Dataset Statistics")
    print("=" * 60)
    
    print(f"\nBasic Counts:")
    print(f"  Total Messages: {stats['total_messages']:,}")
    if stats['attack_messages'] > 0:
        print(f"  Normal Messages: {stats['normal_messages']:,}")
        print(f"  Attack Messages: {stats['attack_messages']:,}")
        print(f"  Attack Percentage: {stats['attack_percentage']:.2f}%")
        print(f"  Class Imbalance Ratio: {stats['class_imbalance_ratio']:.2f}:1")
    
    print(f"\nTemporal Statistics:")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Messages per Second: {stats['messages_per_second']:.2f}")
    
    print(f"\nCAN ID Statistics:")
    print(f"  Unique CAN IDs: {stats['unique_can_ids']}")
    if stats['most_common_id']:
        print(f"  Most Common ID: {stats['most_common_id']}")
        print(f"    Count: {stats['most_common_id_count']:,} ({stats['most_common_id_percentage']:.2f}%)")
    
    print(f"\nData Field Statistics:")
    print(f"  Messages with Data: {stats['messages_with_data']:,}")
    print(f"  Messages without Data: {stats['messages_without_data']:,}")
    print("=" * 60)

def load_data(vehicle, dtype, labeled):
    filename = "data\\" + vehicle + "-" + dtype + ("" if not labeled else "-labeled") + ".csv"
    try:
        with open(filename) as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                data.append(row)
        return np.array(data)
    except FileNotFoundError:
        return -1

def get_id_distribution(data):
    assert(0), "not currently supported"
    # ids = data[1:, 1]
    # unique_ids, counts = np.unique(ids, return_counts=True)
    # return dict(zip(unique_ids, counts))

def naive_feature_extract(raw_data, fe, N, labeled):
    dataset = []
    labels = []
    count = 0
    attack = 0
    curr_sample = []
    curr_t0 = math.trunc(float(raw_data[1,0]) * 100_000)
    for can_message in raw_data[2:]:
        if labeled and int(can_message[3]) == 1:
            attack = 1
        deltat = math.trunc(float(can_message[0])* 100_000) - curr_t0
        curr_sample.append(deltat)
        curr_sample.append(int(can_message[1], 16)) # CAN ID
        if (fe == feature_extraction.NAIVE):
            data_field = can_message[2] if can_message[2] else '0'
            curr_sample.append((int(data_field, 16) & 0xFF000000) >> 24) # CAN byte 0
            curr_sample.append((int(data_field, 16) & 0x00FF0000) >> 16) # CAN byte 1
            curr_sample.append((int(data_field, 16) & 0x0000FF00) >> 8) # CAN byte 2
            curr_sample.append(int(data_field, 16) & 0x000000FF) # CAN byte 3
        count = (count + 1) % N
        if (count == 0):
            # this input is done
            dataset.append(curr_sample)
            if (labeled):
                labels.append(attack)
            # reset variables
            attack = 0
            curr_t0 = math.trunc(float(can_message[0]) * 100_000)
            curr_sample = []
    return np.array(dataset), np.array(labels)

def statistical_feature_extract(raw_data, N, labeled):
    assert(0), "statistical feature extraction not yet implemented"
    return -1


def fetch_dataset(conf):
    assert isinstance(conf, Config), "Fetch dataset needs config class."
    # check if it is already present in the data folder and load data from csv
    raw_data = load_data(conf.vehicle, conf.dtype, conf.labeled)
    # if not, fetch from bitbucket, then load from csv
    if (raw_data is -1):
        download_can_train_and_test_dataset(conf.vehicle,  conf.dtype, conf.labeled)
        raw_data = load_data(conf.vehicle, conf.dtype, conf.labeled)
        assert raw_data != -1, "download or data load of csv failed. Check filenames."
    
    # perform feature extraction
    if conf.fe == feature_extraction.NAIVE or conf.fe == feature_extraction.NO_CAN_DATA:
        dataset, labels = naive_feature_extract(raw_data, conf.fe, conf.N, conf.labeled)
    elif conf.fe == feature_extraction.STATISTICAL:
        dataset, labels = statistical_feature_extract(raw_data, conf.N, conf.labeled)
    else:
        assert(0), "feature extraction selection invalid or not implemented"
    
    # normalize
    scaler = StandardScaler()
    normalized_dataset = scaler.fit_transform(dataset)

    # pack into tensor and return
    tensor_dataset = torch.tensor(normalized_dataset, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.int8)

    return tensor_dataset, tensor_labels

def generate_stats_csv(vehicle, dtypes, labeled_flags, output_filename="dataset_statistics.csv"):
    """
    Generate a CSV file with statistics for multiple datasets.
    
    Args:
        vehicle: Vehicle identifier string
        dtypes: List of dataset types (e.g., ['extra-attack-free', 'fuzzing', 'dos'])
        labeled_flags: List of booleans indicating if each dataset has labels
        output_filename: Name of output CSV file
    
    Returns:
        None (writes to CSV file)
    """
    import csv
    
    # Collect stats for all datasets
    all_stats = []
    for dtype, labeled in zip(dtypes, labeled_flags):
        stats = get_stats(vehicle, dtype, labeled)
        if 'error' not in stats:
            stats['vehicle'] = vehicle
            stats['dataset_type'] = dtype
            stats['labeled'] = labeled
            all_stats.append(stats)
        else:
            print(f"Warning: Could not load {vehicle}-{dtype}, skipping...")
    
    if not all_stats:
        print("No valid datasets found. CSV not created.")
        return
    
    # Define column order for CSV
    columns = [
        'vehicle',
        'dataset_type',
        'labeled',
        'total_messages',
        'normal_messages',
        'attack_messages',
        'attack_percentage',
        'class_imbalance_ratio',
        'duration_seconds',
        'messages_per_second',
        'unique_can_ids',
        'most_common_id',
        'most_common_id_count',
        'most_common_id_percentage',
        'messages_with_data',
        'messages_without_data'
    ]
    
    # Write to CSV
    try:
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for stats in all_stats:
                # Handle None values for unlabeled datasets
                row = {col: stats.get(col, 'N/A') for col in columns}
                writer.writerow(row)
        print(f"Statistics saved to {output_filename}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    # Example 2: Custom selection of datasets
    vehicle = "2016-chevrolet-silverado"
    dtypes = ['extra-attack-free', 'combined', 'dos', 'fuzzy', 'gear', 'interval', 'rpm', 'speed', 'standstill']
    labeled = [False, True, True, True, True, True, True, True, True]
    generate_stats_csv(vehicle, dtypes, labeled, "stats.csv")
    # Get stats for training data (attack-free)
    # train_stats = get_stats("2016-chevrolet-silverado", "extra-attack-free", False)
    # print_stats(train_stats)
    
    # # Get stats for test data (with attacks)
    # test_stats = get_stats("2016-chevrolet-silverado", "fuzzy", True)
    # print_stats(test_stats)
    # # download_can_train_and_test_dataset(vehicle, type)
    # data = load_data(vehicle, type)

    # print(data.shape)
    # print(data[:,1:2].shape)
    # print(data[1:5,:])

    # dataset = create_dataset(data, 3)

    # print(dataset[0:2])
    # id_distr = get_id_distribution(data)
    
    # for id in id_distr:
    #     print(str(id) + ": " + str(id_distr[id]))
    # print(len(id_distr))



    # print(id_distr.values())
    # s = pd.Series(id_distr.values())
    # s.plot.density(bw_method='scott', color='blue', linestyle='-', linewidth=2)
    # plt.show()


    # plt.figure(figsize=(10, 6))
    # plt.bar(id_distr.keys(), id_distr.values())
    # plt.xlabel("ID")
    # plt.ylabel("Count")
    # plt.title("ID Distribution")
    # plt.xticks(rotation=45)  # Rotate labels if many IDs
    # plt.tight_layout()
    # plt.show()
