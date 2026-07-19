import torch
from torch.utils.data import Dataset
import numpy as np
from game import DotsAndBoxesGame

class DotsAndBoxesDataset(Dataset):
    def __init__(self, raw_examples):
        """
        raw_examples: A list of tuples (lines, boxes, pi, v)
        """
        self.examples = raw_examples

    def __len__(self):
        # We multiply the apparent length by 8 to simulate the augmented dataset perfectly.
        return len(self.examples) * 8

    def __getitem__(self, idx):
        # base_idx determines which raw game state we are augmenting
        base_idx = idx // 8
        # sym_idx determines which symmetry (0-7) to apply
        sym_idx = idx % 8

        lines, boxes, pi, v = self.examples[base_idx]

        h, v_mat = DotsAndBoxesGame.l_to_h_v(lines)
        aug_p_h, aug_p_v = DotsAndBoxesGame.l_to_h_v(np.asarray(pi))
        aug_boxes = boxes

        # 1. Rotations (0 to 3 times)
        for _ in range(sym_idx % 4):
            h, v_mat = np.flipud(v_mat.T), np.flipud(h.T)
            aug_p_h, aug_p_v = np.flipud(aug_p_v.T), np.flipud(aug_p_h.T)
            aug_boxes = np.flipud(aug_boxes.T)

        # 2. Reflection (if sym_idx >= 4)
        if sym_idx >= 4:
            h, v_mat = np.fliplr(h), np.fliplr(v_mat)
            aug_p_h, aug_p_v = np.fliplr(aug_p_h), np.fliplr(aug_p_v)
            aug_boxes = np.fliplr(aug_boxes)

        # Reconstruct canonical flat arrays
        aug_p = DotsAndBoxesGame.h_v_to_l(h=aug_p_h, v=aug_p_v)
        size = DotsAndBoxesGame.n_lines_to_size(len(lines))

        # Build 4-channel formatted representation
        c1 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c1[:size + 1, :size] = h

        c2 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c2[:size, :size + 1] = v_mat

        c3 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c3[:size, :size] = (aug_boxes == 1).astype(np.float32)

        c4 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c4[:size, :size] = (aug_boxes == -1).astype(np.float32)

        board_state = np.stack([c1, c2, c3, c4])

        return (
            torch.FloatTensor(board_state),
            torch.FloatTensor(aug_p.astype(np.float32)),
            torch.FloatTensor([float(v)])
        )
