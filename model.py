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
        self.fc_policy = nn.Linear(2 * (game_size + 1) * (game_size + 1), self.action_size)

        # Value Head
        self.conv_value = nn.Conv2d(in_channels=args.num_channels, out_channels=1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear((game_size + 1) * (game_size + 1), 64)
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
        device_name = getattr(args, 'device', None)
        if device_name is not None:
            self.device = torch.device(device_name)
        else:
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
        Returns: tuple of average (pi_loss, v_loss, total_loss)
        """
        print(f'Training on {len(examples)} samples on device: {self.device}')
        self.nnet.train()

        batch_size = self.args.batch_size
        epochs = self.args.epochs

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_loss_all = 0.0

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            pi_l, v_l, t_l = self._train_epoch(examples, batch_size)
            total_pi_loss += pi_l
            total_v_loss += v_l
            total_loss_all += t_l
            
        return total_pi_loss / epochs, total_v_loss / epochs, total_loss_all / epochs

    def _train_epoch(self, examples, batch_size):
        np.random.shuffle(examples)
        
        # tqdm progress bar
        batch_count = int(len(examples) / batch_size)
        if batch_count == 0:
            return 0.0, 0.0, 0.0
            
        t = tqdm(range(batch_count), desc='Training Network')
        
        pi_losses = []
        v_losses = []
        total_losses = []

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
            pi_losses.append(l_pi.item())
            v_losses.append(l_v.item())
            total_losses.append(total_loss.item())
            
            t.set_postfix(Loss_pi=l_pi.item(), Loss_v=l_v.item())

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        return sum(pi_losses)/len(pi_losses), sum(v_losses)/len(v_losses), sum(total_losses)/len(total_losses)

    def format_game_state(self, game):
        """Convert a DotsAndBoxes game into the 4-channel tensor expected by the network."""
        from game import DotsAndBoxesGame

        lines = game.get_canonical_lines()
        boxes = game.get_canonical_boxes()
        size = game.SIZE

        h, v_mat = DotsAndBoxesGame.l_to_h_v(lines)

        c1 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c1[:size + 1, :size] = h

        c2 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c2[:size, :size + 1] = v_mat

        c3 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c3[:size, :size] = (boxes == 1).astype(np.float32)

        c4 = np.zeros((size + 1, size + 1), dtype=np.float32)
        c4[:size, :size] = (boxes == -1).astype(np.float32)

        return np.stack([c1, c2, c3, c4], axis=0)

    def policy_value(self, board):
        """Return policy logits and value estimate for a single board state."""
        board = torch.FloatTensor(np.asarray(board, dtype=np.float64)).unsqueeze(0).to(self.device)
        self.nnet.eval()

        with torch.no_grad():
            logits, value = self.nnet(board)

        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def predict(self, board):
        """
        Outputs policy and value for a single board state.
        board: formatted state representation (channels, x, y)
        """
        logits, value = self.policy_value(board)
        probs = np.exp(logits)
        return probs, value

    def select_action(self, board, valid_moves, temperature=1.0):
        """Sample a valid action from the policy distribution."""
        probs, _ = self.predict(board)
        if valid_moves is None:
            valid_moves = list(range(len(probs)))

        valid_probs = np.zeros_like(probs, dtype=np.float64)
        valid_probs[valid_moves] = probs[valid_moves]
        total = valid_probs.sum()
        if total <= 0:
            action = int(valid_moves[0])
        else:
            valid_probs = valid_probs / total
            action = int(np.random.choice(valid_moves, p=valid_probs[valid_moves]))
        return action

    def train_actor_critic(self, episodes, gamma=0.9, entropy_weight=0.01):
        """Simple actor-critic training from direct returns without MCTS."""
        self.nnet.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_loss = 0.0
        steps_seen = 0

        for epoch in range(self.args.epochs):
            for episode in episodes:
                if not episode:
                    continue

                states = [item[0] for item in episode]
                actions = [int(item[1]) for item in episode]
                rewards = [float(item[2]) if len(item) > 2 else 0.0 for item in episode]

                returns = []
                running_return = 0.0
                for reward in reversed(rewards):
                    running_return = reward + gamma * running_return
                    returns.append(running_return)
                returns.reverse()

                for state, action, target_return in zip(states, actions, returns):
                    board = torch.FloatTensor(np.asarray(state, dtype=np.float64)).unsqueeze(0).to(self.device)
                    action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
                    target = torch.tensor([target_return], dtype=torch.float32, device=self.device)

                    self.optimizer.zero_grad()
                    logits, value = self.nnet(board)

                    value_pred = value.squeeze(-1)
                    log_prob = logits[:, action_tensor].squeeze()
                    advantage = target - value_pred
                    actor_loss = -(log_prob * advantage.detach()).mean()
                    critic_loss = F.mse_loss(value_pred, target)
                    entropy = -(torch.exp(logits) * logits).sum(dim=1).mean()
                    loss = actor_loss + critic_loss + entropy_weight * entropy

                    loss.backward()
                    self.optimizer.step()

                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_loss += loss.item()
                    steps_seen += 1

        if steps_seen == 0:
            return 0.0, 0.0, 0.0

        return total_actor_loss / steps_seen, total_critic_loss / steps_seen, total_loss / steps_seen

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