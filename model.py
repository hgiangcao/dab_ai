import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# ==============================================================================
# 1. PyTorch Neural Network Architecture (nn.Module)
# ==============================================================================

class DotsAndBoxesNet(nn.Module):
    """
    AlphaZero Neural Network Architecture for Dots and Boxes.
    Uses Convolutional blocks and Residual layers.
    """
    def __init__(self, game_size, action_size, args):
        super(DotsAndBoxesNet, self).__init__()
        self.game_size = game_size
        self.action_size = action_size
        self.args = args

        # In Dots and Boxes DualRes encoding, the input usually has multiple channels
        # (e.g., horizontal lines, vertical lines, boxes player 1, boxes player 2).
        # We assume 4 input channels for the standard feature planes.
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=args.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResBlock(args.num_channels) for _ in range(args.num_res_blocks)
        ])

        # Policy Head
        self.conv_policy = nn.Conv2d(in_channels=args.num_channels, out_channels=2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * game_size * game_size, self.action_size)

        # Value Head
        self.conv_value = nn.Conv2d(in_channels=args.num_channels, out_channels=1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(game_size * game_size, 64)
        self.fc_value2 = nn.Linear(64, 1)

    def forward(self, s):
        # s: batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))

        for block in self.res_blocks:
            s = block(s)

        # Policy Head processing
        p = F.relu(self.bn_policy(self.conv_policy(s)))
        p = p.view(p.size(0), -1) 
        p = self.fc_policy(p)
        pi = F.log_softmax(p, dim=1)

        # Value Head processing
        v = F.relu(self.bn_value(self.conv_value(s)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))

        return pi, v


class ResBlock(nn.Module):
    """
    Standard AlphaZero Residual Block.
    """
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


# ==============================================================================
# 2. Network Wrapper (Train, Predict, Save, Load)
# ==============================================================================

class NNetWrapper:
    """
    Wrapper class connecting the DotsAndBoxes engine to the PyTorch neural network.
    Handles device placement, tensor conversion, training loops, and predictions.
    """
    def __init__(self, game, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dimensions based on game (assuming n x n boxes)
        self.board_x, self.board_y = game.getBoardSize() if hasattr(game, 'getBoardSize') else (game.SIZE, game.SIZE)
        self.action_size = game.getActionSize() if hasattr(game, 'getActionSize') else game.N_LINES

        self.nnet = DotsAndBoxesNet(self.board_x, self.action_size, args).to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    def train(self, examples):
        """
        Trains the neural network using self-play generated examples.
        examples: list of (board_state, pi, v)
        """
        print(f'Training on {len(examples)} samples on device: {self.device}')
        self.nnet.train()

        batch_size = self.args.batch_size
        epochs = self.args.epochs

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            self._train_epoch(examples, batch_size)

    def _train_epoch(self, examples, batch_size):
        np.random.shuffle(examples)
        
        # tqdm progress bar
        batch_count = int(len(examples) / batch_size)
        t = tqdm(range(batch_count), desc='Training Network')

        for i in t:
            sample_ids = np.random.randint(len(examples), size=batch_size)
            boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            
            # Format inputs. Ensure shape matches the expected Conv2d input (batch, channels, x, y)
            boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
            target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

            # predict
            out_pi, out_v = self.nnet(boards)
            
            # calculate losses
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v = self.loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

            # record loss
            t.set_postfix(Loss_pi=l_pi.item(), Loss_v=l_v.item())

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def predict(self, board):
        """
        Outputs policy and value for a single board state.
        board: formatted state representation (channels, x, y)
        """
        board = torch.FloatTensor(board.astype(np.float64)).unsqueeze(0).to(self.device)
        self.nnet.eval()
        
        with torch.no_grad():
            pi, v = self.nnet(board)

        # Return probability distribution over actions and scalar value
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """Cross-entropy loss for the policy head"""
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """Mean squared error loss for the value head"""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])