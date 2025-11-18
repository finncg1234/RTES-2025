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


if __name__ == "__main__":
    # download_can_train_and_test_dataset(vehicle, type)
    data = load_data(vehicle, type)

    print(data.shape)
    print(data[:,1:2].shape)
    print(data[1:5,:])

    dataset = create_dataset(data, 3)

    print(dataset[0:2])
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
