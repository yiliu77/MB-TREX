import numpy as np
import torch


def create_pref_validation(all_states, human, device='cuda', samples=10):
    # Create validation set
    pref_states1_batch, pref_states2_batch, pref_actions1_batch, pref_actions2_batch, pref_labels_batch = [], [], [], [], []

    for i in range(samples):
        pref1_index, pref2_index = np.random.randint(0, all_states.shape[1]), np.random.randint(0, all_states.shape[1])
        while pref1_index == pref2_index:
            pref1_index, pref2_index = np.random.randint(0, all_states.shape[1]), np.random.randint(0, all_states.shape[1])
        pref_states1 = all_states[:, pref1_index, ...]
        pref_states2 = all_states[:, pref2_index, ...]
        label = human.query_preference(pref_states1, pref_states2, validate=True)

        pref_states1_batch.append(pref_states1)
        pref_states2_batch.append(pref_states2)
        pref_labels_batch.append(label)

    pref_states1_batch = torch.from_numpy(np.stack(pref_states1_batch, axis=1)).to(device).float()
    pref_states2_batch = torch.from_numpy(np.stack(pref_states2_batch, axis=1)).to(device).float()
    pref_labels_batch = torch.from_numpy(np.stack(pref_labels_batch, axis=0)).to(device).float()

    return pref_states1_batch, pref_states2_batch, pref_labels_batch
