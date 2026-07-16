from multiprocessing import Value
A common and effective pipeline is:

                Expert Game Generation
       (Alpha-Beta / Strong MCTS / Mixed)

                 10k–50k games
                       │
                       ▼
        Extract (state, policy, value)
                       │
                       ▼
              Pretraining Dataset
                       │
             Train 10–20 epochs
                       │
             Save pretrained model
                       │
                       ▼
          AlphaZero Self-Play Training
                       │
        MCTS using pretrained network
                       │
        Generate new self-play games
                       │
        Add to replay buffer
                       │
              Train network
                       │
                  Repeat
Dataset

Each sample should contain:

{
    "state": board_state,
    "policy": target_policy,
    "value": target_value
}
State
state = encode_board(board)

Exactly the same representation used by AlphaZero.

Policy

If using Alpha-Beta directly:

policy = np.zeros(action_size)
policy[best_move] = 1.0

Better:

policy = softmax(search_scores)

or

policy = normalized_visit_counts

A soft target is preferable to a one-hot target.

Value

Final game result from the current player's perspective.

+1   win
 0   draw
-1   lose

Store the final result for every state in the game.

Training loop
for epoch in range(pretrain_epochs):

    for state, policy, value in dataloader:

        pred_policy, pred_value = net(state)

        loss_policy = cross_entropy(pred_policy, policy)
        loss_value  = mse(pred_value, value)

        loss = loss_policy + loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
After pretraining
Load pretrained model
        │
        ▼
Run AlphaZero self-play
        │
        ▼
Collect replay buffer
        │
        ▼
Normal AlphaZero training

Parameter	Value
Expert games	10k–50k
Batch size	256–512
Epochs	10–20
Learning rate	1e-3
Optimizer	AdamW
Value loss	MSE
Policy loss	Cross-Entropy
Weight decay	1e-4