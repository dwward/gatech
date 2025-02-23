
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook.ipynb

import time
from isolation import Board

# Credits if any
# 1) Me
# 2) Myself
# 3) I

class OpenMoveEvalFn:
    def score(self, game, my_player=None):
        """Score the current game state
        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.

        Note:
            If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                game (Board): The board and game state.
                my_player (Player object): This specifies which player you are.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        num_player_moves = len(game.get_player_moves(my_player))
        num_opponent_moves = len(game.get_opponent_moves(my_player))
        return num_player_moves - num_opponent_moves

######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.time_out_buffer = 50 # End recursion when this many millis remaining

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: (int,int): Your best move
        """
        best_move, utility = minimax(self, game, time_left, depth=self.search_depth)
        return best_move

    def utility(self, game, my_turn):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)

###################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE CLASS! ################
###### IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ###########
###################################################################

def minimax(player, game, time_left, depth, my_turn=True):
    """Implementation of the minimax algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you
            need from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer()).
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """

    start_time = time.time()

    if my_turn:
        m, score = max_value(player, game, time_left, depth, True)
    else:
        m, score = min_value(player, game, time_left, depth, False)

    end_time = time.time()
    print("Elapsed time for minimax: {}".format(end_time - start_time))

    return m, score


def max_value(player, game, time_left, depth, my_turn=True):

    if time_left() < player.time_out_buffer:
        return None, float('-inf')

    if depth == 0:
        score = player.utility(game, player)
        position = game.get_player_position(player) if my_turn else game.get_opponent_position(player)
        return position, score

    actions = game.get_player_moves(player)
    #print("My turn, examine all my moves: {}".format(actions))
    value = float('-inf')
    best_move = None
    for move in actions:
        new_board, is_over, winner = game.forecast_move(move)
        m, score = min_value(player, new_board, time_left, depth-1, False)
        if score >= value:
            value = score
            best_move = move
        #print("Scored move {} val: {}".format(m, score))
    return best_move, value

def min_value(player, game, time_left, depth, my_turn=True):

    if time_left() < player.time_out_buffer:
        return None, float('-inf')

    if depth == 0:
        score = player.utility(game, player)
        position = game.get_player_position(player) if my_turn else game.get_opponent_position(player)
        return position, score

    actions = game.get_opponent_moves(player)
    #print("Opponent turn, examine their moves: {}".format(actions))
    value = float('inf')
    best_move = None
    for move in actions:
        new_board, is_over, winner = game.forecast_move(move)
        m, score = max_value(player, new_board, time_left, depth-1, True) # change from f to t
        if score <= value:
            value = score
            best_move = move
        #print("Scored move {} val: {}".format(m, score))
    return best_move, value

######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def alphabeta(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    """Implementation of the alphabeta algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you need
            from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer())
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        alpha (float): Alpha value for pruning
        beta (float): Beta value for pruning
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """
    # TODO: finish this function!
    raise NotImplementedError
    return best_move, val


######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
# tests.name_of_the_test #you can uncomment this line to run your test
################ END OF LOCAL TEST CODE SECTION ######################

class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, my_player=None):
        """Score the current game state.

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args:
            game (Board): The board and game state.
            my_player (Player object): This specifies which player you are.

        Returns:
            float: The current state's score, based on your own heuristic.
        """

        # TODO: finish this function!
        raise NotImplementedError

######################################################################
############ DON'T WRITE ANY CODE OUTSIDE THE CLASS! #################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################