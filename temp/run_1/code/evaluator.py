def play_match(game, player1, player2, num_games):
    half_games = int(num_games / 2)
    p1_wins, p2_wins, draws = 0, 0, 0

    def play_single_game(p1, p2):
        players = {1: p1, -1: p2}
        cur_player = 1
        board = game.getInitBoard()
        
        while game.getGameEnded(board, cur_player) == 0:
            canonical_board = game.getCanonicalForm(board, cur_player)
            action = players[cur_player](canonical_board)
            board, cur_player = game.getNextState(board, cur_player, action)
            
        return cur_player * game.getGameEnded(board, cur_player)

    # First half: player1 moves first
    for _ in range(half_games):
        result = play_single_game(player1, player2)
        if result == 1: p1_wins += 1
        elif result == -1: p2_wins += 1
        else: draws += 1

    # Second half: player2 moves first
    for _ in range(half_games):
        result = play_single_game(player2, player1)
        # Result is from the perspective of player2
        if result == -1: p1_wins += 1
        elif result == 1: p2_wins += 1
        else: draws += 1

    return p1_wins, p2_wins, draws