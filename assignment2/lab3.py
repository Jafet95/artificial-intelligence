# Curso de maestria: Inteligencia artificial (TEC), I Semestre 2019
# Author: Jafet Chaves Barrantes <jafet.a15@gmail.com>

from util import INFINITY

### 1. Escogencia multiple

# 1.1. Dos jugadores computarizados estan jugando un juego. E Jugador MM utiliza minimax
#      para buscar a una profundidad de 6 para decidir sobre una movida. El jugador AB utiliza alpha-beta
#      para buscar a una profundidad de 6.
#      El juego se lleva a cabo sin un limite de tiempo. Cual jugador jugara mejor?
#
#      1. MM jugara mejor que AB.
#      2. AB jugara mejor que  MM.
#      3. Ambos jugaran al mismo nivel de destreza.
ANSWER1 = 3

# 1.2. Dos jugadores computarizados juegan un juego con un limite de tiempo. El jugador MM
# hace busqueda minimax con profundidad iterativa, y el jugador AB hace busqueda alpha-beta
# con profundidad iterativa. Cada uno retorna un resultado despues de haber utilizado
# 1/3 de su tiempo restante. Cual jugador jugara mejor?
#
#      1. MM jugara mejor que AB.
#      2. AB jugara mejor que  MM.
#      3. Ambos jugaran al mismo nivel de destreza.
ANSWER2 = 2

### 2. Connect Four
from connectfour import *
from basicplayer import *
from util import *
import tree_searcher

## Esta seccion contiene lineas ocasionales que puede descomentar para jugar
## el juego interactivamente. Asegurese de re-comentar cuando ha terminado con
## ellos.  Por favor no entregue su tarea con partes de codigo que solicitan
## jugar interactivamente!
## 
## Descomente la siguiente linea para jugar el juego como las blancas:
# ~ run_game(human_player, basic_player)

## Descomente la siguiente linea para jugar como las negras:
#run_game(basic_player, human_player)

## O bien vea a la computadora jugar con si misma:
#run_game(basic_player, basic_player)

## Cambie la siguiente funcion de evaluacion tal que trata de ganar lo mas rapido posible,
## o perder lentamente, cuando decide que un jugador esta destina a ganar.
## No tiene que cambiar como evalua posiciones que no son ganadoras.

def focused_evaluate(board):
    """
    Dado un tablero, returna un valor numerico indicando que tan bueno
    es el tablero para el jugador de turno.
    Un valor de retorno >= 1000 significa que el jugador actual ha ganado;
    Un valor de retorno <= -1000 significa que el jugador actual perdio
    """
    score = 0
    #Evalua si el jugador actual ha hecho un conecta 4
    if board.is_win() == board.get_current_player_id():
        score = 1000 - board.num_tokens_on_board()
    #Evalua si el otro jugador ha hecho un conecta 4
    elif board.is_win() == board.get_other_player_id():
        score = -1000 + board.num_tokens_on_board()
    #Si nadie ha ganado calcule el puntaje del tablero para el jugador
    else:
        score = board.longest_chain(board.get_current_player_id()) * 7
        #Prefiere poner sus piezas en el centro del tablero
        for row in range(6):
            for col in range(7):
                if board.get_cell(row, col) == board.get_current_player_id():
                    score -= abs(3-col)
                elif board.get_cell(row, col) == board.get_other_player_id():
                    score += abs(3-col)
    return score


## Crea una funcion "jugador" que utiliza la funcion focused_evaluate function
quick_to_win_player = lambda board: minimax(board, depth=4,
                                            eval_fn=focused_evaluate)

## Puede probar su nueva funcion de evaluacion descomentando la siguiente linea:
# ~ run_game(basic_player, quick_to_win_player)
# ~ run_game(quick_to_win_player, basic_player)
# ~ run_game(basic_player, basic_player)

## Escriba un procedimiento de busqueda alpha-beta-search que actua como el procedimiento minimax-search
## pero que utiliza poda alpha-beta para evitar buscar por malas ideas
## que no pueden mejorar el resultado. El tester revisara la poda
## contando la cantidad de evaluaciones estaticas que hace
##
## Puede utilizar el minimax() que se encuentra basicplayer.py como ejemplo.

##Alpha-beta pruning es muy similar a minimax con la optimizacion:
##"If m is better than n for Player, we will never get to n in play"
#alpha = the value of the best (i.e., highest-value) choice we have found so
#far at any choice point along the path for MAX.
#beta = the value of the best (i.e., lowest-value) choice we have found so
#far at any choice point along the path for MIN.
#La busqueda debe retornar el numero de columna en la cual debe
#agregar la ficha (best move)

def max_value(board, depth, eval_fn, get_next_moves_fn, is_terminal_fn, alpha, beta):
    if is_terminal_fn(depth, board):
        return eval_fn(board)

    val = NEG_INFINITY

    for move, new_board in get_next_moves_fn(board):
        val = max(val, min_value(new_board, depth-1, eval_fn, get_next_moves_fn, is_terminal_fn, alpha, beta))
        if val >= beta:
            return val
        alpha = max(val,alpha)
    return val

def min_value(board, depth, eval_fn, get_next_moves_fn, is_terminal_fn, alpha, beta):
    if is_terminal_fn(depth, board):
        return eval_fn(board)

    val = INFINITY

    for move, new_board in get_next_moves_fn(board):
        val = min(val, max_value(new_board, depth-1, eval_fn, get_next_moves_fn, is_terminal_fn, alpha, beta))
        if val <= alpha:
            return val
        beta = min(val,beta)
    return val

def alpha_beta_search(board, depth,
                      eval_fn,
                      # NOTA: usted debe utilizar get_next_moves_fn cuando genera
                      # configuraciones de proximos tableros, y utilizar is_terminal_fn para
                      # revisar si el juego termino.
                      # Las funciones que por defecto se asignan aqui funcionarar 
                      # para connect_four.
                      get_next_moves_fn=get_all_next_moves,
                      is_terminal_fn=is_terminal):

    best_move = None
    best_val = None
    best_board = None
    alpha = None
    beta = None

    #Para los movimientos posibles en el tablero actual
    for move, new_board in get_next_moves_fn(board):
        #Tecnica de negmax (maximizar siempre desde el jugador)
        maxmin = -1 * max_value(new_board, depth-1, eval_fn, get_next_moves_fn, is_terminal_fn, alpha, beta)
        if best_move == None or maxmin > best_val:
            best_move = move
            best_val = maxmin
            best_board = new_board

    return best_move

## Ahora deberia ser capaz de buscar al doble de profundidad en la misma cantidad de tiempo.
## (Claro que este jugador alpha-beta-player no funcionara hasta que haya definido
## alpha-beta-search.)
alphabeta_player = lambda board: alpha_beta_search(board,
                                                   depth=8,
                                                   eval_fn=focused_evaluate)

## Este jugador utiliza profundidad iterativa, asi que le puede ganar mientras hace uso 
## eficiente del tiempo:
ab_iterative_player = lambda board: \
    run_search_function(board,
                        search_fn=alpha_beta_search,
                        eval_fn=focused_evaluate, timeout=5)
# ~ run_game(human_player, alphabeta_player)

## Finalmente, aqui debe crear una funcion de evaluacion mejor que focused-evaluate.
## By providing a different function, you should be able to beat
## simple-evaluate (or focused-evaluate) while searching to the
## same depth.

def better_evaluate(board):
    score = 0
    a_columns = 0
    a_rows = 0
    na_columns = 0
    na_rows = 0
    vacias=42
    if board.is_win() == board.get_current_player_id():
        score = INFINITY
    elif board.is_win() == board.get_other_player_id():
        score = NEG_INFINITY
    else:
        #Bonuses
        score += board.longest_chain(board.get_current_player_id())*10
        list_current_player=list(board.chain_cells(board.get_current_player_id()))
        for chain in list_current_player:
            if len(chain) >= 3:
                score += 30
        for row in range(6):
            for col in range(7):
                if board.get_cell(row, col) == board.get_current_player_id():
                    na_columns += abs(3-col)*2
                    na_rows += 2
                    vacias -= 1
                if board.get_cell(row, col) == board.get_other_player_id():
                    na_columns += abs(3-col)
                    na_rows += 1
                    vacias -= 1
                if board.get_cell(row, col) == 0:
                    a_columns += abs(3-col)*2
                    a_rows += 1
                    if vacias <= 28 and (col == 7):
                        score += INFINITY
            score += (a_columns + a_rows)-(na_columns+na_rows)

    return score

# Comente esta linea una vez que ha implementado completamente better_evaluate
# ~ better_evaluate = memoize(basic_evaluate)

# Descomente esta linea para hacer que su better_evaluate corra mas rapido.
better_evaluate = memoize(better_evaluate)

# Para el debugging: Cambie este if-guard a True, para hacer unit-test
# de su funcion better_evaluate.
if False:
    board_tuples = (( 0,0,0,0,0,0,0 ),
                    ( 0,0,0,0,0,0,0 ),
                    ( 0,0,0,0,0,0,0 ),
                    ( 0,2,2,1,1,2,0 ),
                    ( 0,2,1,2,1,2,0 ),
                    ( 2,1,2,1,1,1,0 ),
                    )
    board_tuples2 = (( 0,0,0,0,0,0,0 ),
                    ( 0,0,0,0,0,0,0 ),
                    ( 0,0,0,0,0,0,0 ),
                    ( 0,2,2,1,1,2,0 ),
                    ( 0,2,1,2,1,2,0 ),
                    ( 2,1,2,1,1,1,0 ),
                    )
    test_board_1 = ConnectFourBoard(board_array = board_tuples,
                                    current_player = 1)
    test_board_2 = ConnectFourBoard(board_array = board_tuples,
                                    current_player = 2)
    # better evaluate de jugador 1
    print "%s => %s" %(test_board_1, better_evaluate(test_board_1))
    # better evaluate de jugador 2
    print "%s => %s" %(test_board_2, better_evaluate(test_board_2))

## Un jugador que utiliza alpha-beta y better_evaluate:
your_player = lambda board: run_search_function(board,
                                                search_fn=alpha_beta_search,
                                                eval_fn=better_evaluate,
                                                timeout=5)

# ~ your_player = lambda board: alpha_beta_search(board, depth=8,
                                              # ~ eval_fn=better_evaluate)

## Descomente para ver su jugador jugar un juego:

## Descomente esto (o corralo en una ventana) para ver como le va 
## en el torneo que sera evaluado.
# ~ run_game(human_player, your_player)
# ~ run_game(basic_player, your_player)
# ~ run_game(quick_to_win_player, your_player)
# ~ run_game(human_player, your_player)
# ~ run_game(your_player, your_player)

## Estas funciones son utilizadas por el tester, por favor no las modifique!
def run_test_game(player1, player2, board):
    assert isinstance(globals()[board], ConnectFourBoard), "Error: can't run a game using a non-Board object!"
    return run_game(globals()[player1], globals()[player2], globals()[board])
    
def run_test_search(search, board, depth, eval_fn):
    assert isinstance(globals()[board], ConnectFourBoard), "Error: can't run a game using a non-Board object!"
    return globals()[search](globals()[board], depth=depth,
                             eval_fn=globals()[eval_fn])

## Esta funcion corre su implementacion de alpha-beta utilizando un arbol de busqueda 
## en vez de un juego en vivo de connect four.   Esto sera mas facil de debuggear.
def run_test_tree_search(search, board, depth):
    return globals()[search](globals()[board], depth=depth,
                             eval_fn=tree_searcher.tree_eval,
                             get_next_moves_fn=tree_searcher.tree_get_next_move,
                             is_terminal_fn=tree_searcher.is_leaf)
    
## Quiere utilizar su codigo en un torneo con otros estudiantes? Vea 
## la descripcion en el enunciado de la tarea. El torneo es opcional
## y no tiene efecto en su nota
COMPETE = (False)

## The standard survey questions.
HOW_MANY_HOURS_THIS_PSET_TOOK = "6"
WHAT_I_FOUND_INTERESTING = "La implementacion del better evaluate"
WHAT_I_FOUND_BORING = "Nada"
NAME = "Jafet Chaves"
EMAIL = "jafet.a15@gmail.com"

