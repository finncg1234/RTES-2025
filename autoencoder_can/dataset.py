import urllib.request
import csv
import numpy as np
import torch
import math

vehicle = "2016-chevrolet-silverado"
type = "extra-attack-free"
def download_can_train_and_test_dataset(vehicle, type):
    url = "https://bitbucket.org/brooke-lampe/can-dataset/raw/c7e146cc75a5bac5d79e2a064e300e2276d24c51/" \
        + vehicle + "/" + type + "/" + type + "-1.csv"
    output_filename = "data\\" + vehicle + "-" + type + ".csv"

    try:
        urllib.request.urlretrieve(url, output_filename)
        print(f"File downloaded successfully to {output_filename}")
    except Exception as e:
        print(f"Error during download: {e}")

def load_data(vehicle, type):
    filename = "data\\" + vehicle + "-" + type + ".csv"
    with open(filename) as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    return np.array(data)

def get_id_distribution(data):
    ids = data[1:, 1]
    unique_ids, counts = np.unique(ids, return_counts=True)
    return dict(zip(unique_ids, counts))

def create_dataset(data, N):
    dataset = []
    count = 0
    curr_sample = []
    curr_t0 = math.trunc(float(data[1,0]) * 100_000)
    for can_message in data[2:]:
        deltat = math.trunc(float(can_message[0])* 100_000) - curr_t0
        curr_sample.append(deltat)
        curr_sample.append(int(can_message[1], 16))
        count = (count + 1) % N
        if (count == 0):
            # this input is done
            dataset.append(curr_sample)

            # reset variables
            curr_t0 = math.trunc(float(can_message[0]) * 100_000)
            curr_sample = []
    
    return torch.tensor(dataset, dtype=torch.int32)

    


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
