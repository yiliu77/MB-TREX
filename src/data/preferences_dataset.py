from torch.utils import data


class PreferencesDataset(data.Dataset):
    def __init__(self, sequences1, actions1, sequences2, actions2, labels):
        self.sequences1 = sequences1
        self.actions1 = actions1
        self.sequences2 = sequences2
        self.actions2 = actions2
        self.labels = labels

    def __len__(self):
        return len(self.sequences1)

    def __getitem__(self, index):
        return self.sequences1[index], self.actions1[index], self.sequences2[index], self.actions2[index], self.labels[index]
