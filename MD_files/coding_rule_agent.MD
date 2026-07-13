 	Rules
In the first league the board is a 2x2 square grid of boxes.

The two players are named 'A' and 'B'. Player 'A' plays first.

The boxes are named with a one letter column and a one number row, for example 'A1'. A is the leftmost column 1 is the downmost row.

Each box has 4 sides named 'L'=left, 'T'=top, 'R'=right and 'B'=bottom. Note that most sides are shared by two boxes. For example A1 R and B1 L designate the same box side.

Player 'A' plays first.
Each player in turn draws a free box side (not already drawn by him or his opponent).
If the box side closes one or two boxes the player gains ownership of these boxes and the same player plays the next turn. And so on until the player does not close a box.
The game terminates when there are no more vacant box side to be drawn.
Victory Conditions
Own more boxes than your opponent.
Note that a draw is not feasible if the total number of boxes is odd.
Loss Conditions
Own less boxes than your opponent.
You do not respond in due time or output an invalid command.
 	Game Input
The program must first read the game state input data from standard input. Then, provide to the standard output one line with the player's move.
Initial input
First line: boardSize: the number of rows and columns on the board.
Second line: playerId: the ID of the player. 'A' or 'B'.
Input per turn
First line: playerScore opponentScore: the number of boxes that you and your opponent actually own.

Second line: numBoxes: the number of playable boxes.

Next numBoxes lines: box sides: two string. The first string is the box name, for example "A1". The second string lists the playable sides for that box. "LTRB" for example if the four sides of the box are free .

Output
A single line: the box name and side to draw. This can be followed by an optional message to display introduced by the 'MSG' keyword. Example: "A1 R MSG Hello".
Constraints
Allotted response time to output is 1 s for the first turn, then 100 ms for the other turns.

----
python template
import sys
import math

# Own more boxes than your opponent!

board_size = int(input())  # The size of the board.
player_id = input()  # The ID of the player. 'A'=first player, 'B'=second player.

# game loop
while True:
    # player_score: The player's score.
    # opponent_score: The opponent's score.
    player_score, opponent_score = [int(i) for i in input().split()]
    num_boxes = int(input())  # The number of playable boxes.
    for i in range(num_boxes):
        # box: The ID of the playable box.
        # sides: Playable sides of the box.
        box, sides = input().split()

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # <box> <side> [MSG Optional message]
    print("A1 B MSG bla bla bla...")
