from torch.utils.data import Dataset
import torch


class Data(Dataset):

    def __init__(self, data):
        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = data

    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        example, label = self._samples[index]

        return example, label

    def __len__(self):
        """Total number of samples"""
        return len(self._samples)

    def get_sample_by_id(self, id_):
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]


class BlackBoxWrapper():
    def __init__(self, model,
                 num_players,
                 device):
        self.model = model
        self.num_players = num_players
        # self.scaler = scaler
        self.device = device

    def __call__(self, x, S):
        '''
        Evaluate a black-box model.
        Args:
          x: input examples.
          S: coalitions.
        '''
        # x = self.scaler.transform(x)
        x = x * S

        # x = self.scaler.inverse_transform(x)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.reshape((x.shape[0], self.num_players))
        values = self.model(x)

        return values


def transform_to_tensors(data, labels, num_features):
    data_array = data.values  # Convert DataFrame to NumPy array
    tensor_data = torch.FloatTensor(data_array).view(-1, num_features)
    print(tensor_data.shape)

    tensor_labels = torch.tensor(labels.values, dtype=torch.float)  # .view(-1, 1)
    print(tensor_labels.shape)

    list_of_tensors = list(zip(tensor_data, tensor_labels))
    return list_of_tensors


def name_index(data):
    index_to_name = {i: n for i, n in enumerate(data.columns)}
    name_to_index = {n: i for i, n in enumerate(data.columns)}

    return index_to_name, name_to_index
