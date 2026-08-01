import torch
from torch.utils.data import Dataset
import numpy as np
from game import DotsAndBoxesGame

class DotsAndBoxesDataset(Dataset):
    def __init__(self, raw_examples):
        """
        raw_examples: A list of tuples (lines, boxes, pi, v)
        No rotation/mirror augmentation — keeps original 100,000 samples as-is.
        """
        # Convert Python lists to contiguous numpy arrays to prevent massive RAM overhead
        # and Pickling crashes when DataLoader spawns multiprocessing workers.
        lines_list, boxes_list, pi_list, v_list = zip(*raw_examples)

        self.lines = np.array(lines_list, dtype=np.float32)
        self.boxes = np.array(boxes_list, dtype=np.float32)
        self.pis   = np.array(pi_list,   dtype=np.float32)
        self.vs    = np.array(v_list,    dtype=np.float32)
        self.length = len(raw_examples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        lines = self.lines[idx]
        boxes = self.boxes[idx]
        pi    = self.pis[idx]
        v     = self.vs[idx]

        h, v_mat = DotsAndBoxesGame.l_to_h_v(lines)
        size = DotsAndBoxesGame.n_lines_to_size(len(lines))

        # Build 4-channel board representation (no augmentation)
        c1 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c1[:size + 1, :size] = h

        c2 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c2[:size, :size + 1] = v_mat

        c3 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c3[:size, :size] = (boxes == 1).astype(np.float32)

        c4 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c4[:size, :size] = (boxes == -1).astype(np.float32)

        board_state = np.stack([c1, c2, c3, c4])

        return (
            torch.FloatTensor(board_state),
            torch.FloatTensor(pi.astype(np.float32)),
            torch.tensor(float(v), dtype=torch.float32)
        )
