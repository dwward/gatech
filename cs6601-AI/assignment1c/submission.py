#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook.ipynb

import time
import numpy

from isolation import Board


# Credits if any
# 1) Me
# 2) Myself
# 3) I

class OpenMoveEvalFn:
    def score(self, game, my_player=None):
        num_player_moves = len(game.get_player_moves(my_player))
        num_opponent_moves = len(game.get_opponent_moves(my_player))
        return num_player_moves - num_opponent_moves


######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################


class CustomEvalFn:
    def __init__(self):
        pass

    def area_score(self, pos, game):
        score = 0

        # fat
        bubble = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 1),
                  (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                  (0, -2), (0, -1), (0, 1), (0, 2),
                  (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                  (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
                  ]

        # skinny
        # bubble = [(-2, -1), (-2, 0), (-2, 1),
        #          (-1, -1), (-1, 0), (-1, 1),
        #          (0, -1), (0, 1),
        #          (1, -1), (1, 0), (1, 1),
        #          (2, -1), (2, 0), (2, 1),
        #          ]
        for s in bubble:
            spot = numpy.add(pos, s)
            row, col = spot[0], spot[1]
            if game.space_is_open(row, col):
                score = score + int(game.is_spot_open(row, col))
        return score

    def score(self, game, my_player=None):
        num_player_moves = len(game.get_player_moves(my_player))
        num_opponent_moves = 1.7 * len(game.get_opponent_moves(my_player))
        my_area_score = 0
        opp_area_score = 0

        # if game.move_count > 30:
        #     my_pos = game.get_player_position(my_player)
        #     my_area_score = self.area_score(my_pos, game)
        #     opp_pos = game.get_opponent_position(my_player)
        #     opp_area_score = self.area_score(opp_pos, game)

        # game.get_player_position(my_player)

        # if game.move_count < 8:
        #     num_opponent_moves = num_opponent_moves * 2
        # if game.move_count > 30:
        #     num_opponent_moves = num_opponent_moves * 0.7
        return (num_player_moves + my_area_score) - (num_opponent_moves + opp_area_score)


######################################################################
############ DON'T WRITE ANY CODE OUTSIDE THE CLASS! #################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################


class CustomPlayer:

    def __init__(self, search_depth=9999, eval_fn=CustomEvalFn(), use_open_moves=False):

        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.time_out_buffer = 70  # End recursion when this many millis remaining
        self.moved = False
        self.use_open_moves = use_open_moves
        self.history = {}
        self.center = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4)]
        # self.top_bottom_edges = [(0, 2), (0, 3), (0, 4), (6, 2), (6, 3), (6, 4)]
        # self.left_right_edges = [(2, 0), (3, 0), (4, 0), (2, 6), (3, 6), (4, 6)]
        self.top_bottom_edges = [(0, 3), (6, 3)]
        self.left_right_edges = [(3, 0), (3, 6)]
        self.entomb_axis = None

    def move(self, game, time_left):
        curr_dep = 0
        while True:
            try:
                best_move, utility = alphabeta(self, game, time_left, depth=curr_dep,
                                               alpha=-999,
                                               beta=999, my_turn=True, history=self.history)
            except TimeoutError:
                break
            curr_dep = curr_dep + 1
        self.history.clear()
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

    def max_value(player, game, time_left, depth, my_turn=True):

        if time_left() < player.time_out_buffer:
            raise TimeoutError()

        if depth == 0:
            score = player.utility(game, player)
            position = game.get_player_position(player) if my_turn else game.get_opponent_position(player)
            return position, score

        actions = game.get_player_moves(player)

        # print("My turn, examine all my moves: {}".format(actions))
        value = -999
        best_move = actions[0]

        try:
            for move in actions:
                new_board, is_over, winner = game.forecast_move(move)
                m, score = min_value(player, new_board, time_left, depth - 1, False)
                if score > value:
                    value = score
                    best_move = move
                # print("Scored move {} val: {}".format(m, score))
        except TimeoutError:
            pass

        return best_move, value

    def min_value(player, game, time_left, depth, my_turn=False):

        if time_left() < player.time_out_buffer:
            raise TimeoutError()

        if depth == 0:
            score = player.utility(game, player)
            position = game.get_player_position(player) if my_turn else game.get_opponent_position(player)
            return position, score

        actions = game.get_opponent_moves(player)

        # print("Opponent turn, examine their moves: {}".format(actions))
        value = 999
        best_move = actions[0]

        try:
            for move in actions:
                new_board, is_over, winner = game.forecast_move(move)
                m, score = max_value(player, new_board, time_left, depth - 1, True)  # change from f to t
                if score < value:
                    value = score
                    best_move = move
                # print("Scored move {} val: {}".format(m, score))
        except TimeoutError:
            pass

        return best_move, value

    #    start_time = time.time()
    if my_turn:
        m, score = max_value(player, game, time_left, depth, True)
    else:
        m, score = min_value(player, game, time_left, depth, False)
    #     end_time = time.time()
    #     print("Time to move for minimax: {}".format(end_time - start_time))

    return m, score


######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
# tests.beatRandom(CustomPlayer)
# tests.minimaxTest(CustomPlayer, minimax)
################ END OF LOCAL TEST CODE SECTION ######################

def alphabeta(player, game, time_left, depth, alpha, beta, my_turn=True, history=None):

    if time_left() < player.time_out_buffer:
        raise TimeoutError()

    # ----------------- TERMINAL NODE ----------------------
    if depth == 0 or len(game.get_player_moves(player)) == 0:
        score = player.utility(game, player)
        position = game.get_player_position(player) if my_turn else game.get_opponent_position(player)
        return position, score

    active_players_queen = game.get_active_players_queen()
    killer_move = False
    board_h = hash(game)

    utility_scores = []

    # ----------------- MAXIMIZE ----------------------
    if my_turn:
        actions = game.get_player_moves(player)
        best_value = alpha

        # Calculate move utility, order moves and store in cache
        if board_h not in history:
            for move in actions:
                new_board, is_over, winner = game.forecast_move(move)
                utility_scores.append(player.utility(new_board, True))
                history[(board_h, move)] = new_board, is_over, winner
            actions = [x for _, x in sorted(zip(utility_scores, actions), reverse=True)]
            history[board_h] = actions
        else:
            actions = history[board_h]
        best_move = actions[0]

        # Get best score from next level down
        for move in actions:
            new_board, is_over, winner = history[(board_h, move)]

            if winner == active_players_queen:
                return move, 999

            m, curr_score = alphabeta(player, new_board, time_left, depth - 1, alpha, beta, False, history)

            if curr_score > best_value:
                best_value = curr_score
                best_move = move

            if best_value >= beta:
                killer_move = True
                break

            alpha = max(alpha, best_value)

        if killer_move:
            actions.remove(best_move)
            actions.insert(0, best_move)
            history[board_h] = actions

        return best_move, best_value

    # ----------------- MINIMIZE ----------------------
    else:
        actions = game.get_opponent_moves(player)
        best_value = beta

        # Calculate move utility, order moves and store in cache
        if board_h not in history:
            for move in actions:
                new_board, is_over, winner = game.forecast_move(move)
                utility_scores.append(player.utility(new_board, True))
                history[(board_h, move)] = new_board, is_over, winner
            actions = [x for _, x in sorted(zip(utility_scores, actions), reverse=False)]
            history[board_h] = actions
        else:
            actions = history[board_h]
        best_move = actions[0]

        # Get best score from next level down
        for move in actions:
            new_board, is_over, winner = history[(board_h, move)]

            # if winner == active_players_queen:
            #     return move, -999

            m, curr_score = alphabeta(player, new_board, time_left, depth - 1, alpha, beta,
                                 True, history)
            if curr_score < best_value:
                best_value = curr_score
                best_move = move

            if best_value <= alpha:
                return best_move, best_value
            beta = min(beta, best_value)

        return best_move, best_value

######################################################################
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
######## IF YOU WANT TO CALL OR TEST IT CREATE A NEW CELL ############
######################################################################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
# tests.beat_minimax(CustomPlayer)
# tests.minimaxTest(CustomPlayer, minimax)
################ END OF LOCAL TEST CODE SECTION ######################
