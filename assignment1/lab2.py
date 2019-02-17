#
# Curso de maestria: Inteligencia artificial (TEC)
# Author: Jafet Chaves Barrantes <jafet.a15@gmail.com>

# Sus respuestas para las preguntas falso y verdadero deben tener la siguiente forma.
# Sus respuestas deben verse como las dos siguientes:
#ANSWER1 = True
#ANSWER1 = False

# 1: Falso o Verdadero - busqueda Hill Climbing garantiza encontrar una respuesta
#    si es que la hay
ANSWER1 = False

# 2: Falso o Verdadero - busqueda Best-first encontrara una ruta optima
#    (camino mas corto).
ANSWER2 = False

# 3: Falso o Verdadero - Best-first y Hill climbing hacen uso de el
#    valor de la heuristica de los nodos.
ANSWER3 = True

# 4: Falso o Verdadero - A* utiliza un conjunto extendido de nodos
ANSWER4 = True

# 5: Falso o Verdadero - Anchura primero esta garantizado a encontrar un
#    camino con el minimo numero de nodos posible
ANSWER5 = True

# 6: Falso o Verdadero - El Branch and bound regular utiliza valores de
#    la heuristica para acelerar la busqueda de un camino optimo
ANSWER6 = False

# Import the Graph data structure from 'search.py'
# Refer to search.py for documentation
from search import Graph

# Implemente estos y los puede revisar con el modulo tester

#Breadth-first search, la agenda se administra como una cola
#Entrada:
#    graph: el grafo
#    start: nombre del nodo de inicio
#    goal: nombre del nodo objetivo
#Salida:
#    lista: goal_path (lista con el camino al nodo objetivo) o vacia
def bfs(graph, start, goal):
    #Se crea la agenda con el primer nodo
    agenda = [(start,)]
    #Ya se alcanzo el objetivo, el nodo de inicio es el objetivo
    if start == goal:
        print('Start node is the same as the goal node')
        return [start]
    #Mientras la agenda no este vacia
    while len(agenda) > 0:
        #Crea un nuevo camino
        new_paths = []
        #El primer camino en la agenda es el que hay que extender (FIFO)
        current_path = agenda[0]
        #Actualizar la agenda quitando los caminos explorados
        agenda.remove(current_path)
        #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
        current_node = current_path[-1]
        #Obtenga la lista de nodos conectados al nodo a extender
        new_nodes = graph.get_connected_nodes(current_node)
        #Elimine los nodos repetidos del camino actual
        if len(current_path) > 1:
            new_nodes = [ nodes for nodes in new_nodes if nodes not in current_path]
        #Revise si el objetivo esta en los nodos adyacentes al nodo actual
        if goal in new_nodes:
            goal_path = current_path + (goal,)
            #Regrese la lista con el camino al objetivo (backtracking)
            return list(goal_path)
        #Agregue los caminos por explorar
        for nodes in new_nodes:
            new_paths += [ current_path + (nodes,)]
        #Extienda la agenda con los nuevos caminos
        agenda.extend(new_paths)
    print("I couldn't find a path to the goal :(")
    #Retorna una lista vacia si no encuentra un camino al nodo objetivo
    return []

## Si hizo el anterior el siguiente debe ser muy sencillo

#Depth-first search, la agenda se administra como una pila
#Entrada:
#    graph: el grafo
#    start: nombre del nodo de inicio
#    goal: nombre del nodo objetivo
#Salida:
#    lista: goal_path (lista con el camino al nodo objetivo) o vacia
def dfs(graph, start, goal):
    #Se crea la agenda con el primer nodo
    agenda = [(start,)]
    #Ya se alcanzo el objetivo, el nodo de inicio es el objetivo
    if start == goal:
        print('Start node is the same as the goal node')
        return [start]
    #Mientras la agenda no este vacia
    while len(agenda) > 0:
        #Crea un nuevo camino
        new_paths = []
        #El primer camino en la agenda es el que hay que extender (FIFO)
        current_path = agenda[0]
        #Actualizar la agenda quitando los caminos explorados
        agenda.remove(current_path)
        #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
        current_node = current_path[-1]
        #Obtenga la lista de nodos conectados al nodo a extender
        new_nodes = graph.get_connected_nodes(current_node)
        #Elimine los nodos repetidos del camino actual
        if len(current_path) > 1:
            new_nodes = [ nodes for nodes in new_nodes if nodes not in current_path]
        #Revise si el objetivo esta en los nodos adyacentes al nodo actual
        if goal in new_nodes:
            goal_path = current_path + (goal,)
            return list(goal_path)
        #Agregue los caminos por explorar
        for nodes in new_nodes:
            new_paths += [ current_path + (nodes,)]
        #Los nuevos caminos se agregan como una pila (LIFO)
        new_paths.extend(agenda)
        agenda = new_paths
    print("I couldn't find a path to the goal :(")
    #Retorna una lista vacia si no encuentra un camino al nodo objetivo
    return []

## Ahora agregue heuristica a su busqueda
## Hill-climbing puede verse como un tipo de busqueda a profundidad primero
## La busqueda debe ser hacia los valores mas bajos que indica la heuristica

#Retorna los nuevos caminos ordenados ascendentemente
def sort_paths(graph,goal,new_paths):
    heuristic_to_goal_list = []
    sorted_paths = []
    for path in new_paths:
        #Retorna el valor de la heuristica del nodo actual al objetivo
        heuristic_to_goal = graph.get_heuristic(path[-1],goal)
        heuristic_to_goal_list.append([path,heuristic_to_goal])
    #Se ordenan los caminos de acuerdo a la heuristica
    heuristic_to_goal_list=sorted(heuristic_to_goal_list, key=lambda x: x[1], reverse=False)
    for paths in heuristic_to_goal_list:
        #Se recuperan solo los caminos de la lista ordenada de heuristicas
        sorted_paths.append(paths[0])
    return sorted_paths

#Hill-climbing search (greedy local search), similar a dfs pero deben
#ordenarse los nodos primero de acuerdo a la heuristica antes de
#actualizar el LIFO
#Entrada:
#    graph: el grafo
#    start: nombre del nodo de inicio
#    goal: nombre del nodo objetivo
#Salida:
#    lista: goal_path (lista con el camino al nodo objetivo) o vacia
def hill_climbing(graph, start, goal):
    #Se crea la agenda con el primer nodo
    agenda = [(start,)]
    #Ya se alcanzo el objetivo, el nodo de inicio es el objetivo
    if start == goal:
        print('Start node is the same as the goal node')
        return [start]
    #Mientras la agenda no este vacia
    while len(agenda) > 0:
        #Crea un nuevo camino
        new_paths = []
        #El primer camino en la agenda es el que hay que extender
        current_path = agenda[0]
        #Actualizar la agenda quitando los caminos explorados
        agenda.remove(current_path)
        #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
        current_node = current_path[-1]
        #Obtenga la lista de nodos conectados al nodo a extender
        new_nodes = graph.get_connected_nodes(current_node)
        #Elimine los nodos repetidos del camino actual
        if len(current_path) > 1:
            new_nodes = [ nodes for nodes in new_nodes if nodes not in current_path]
        #Revise si el objetivo esta en los nodos adyacentes al nodo actual
        if goal in new_nodes:
            goal_path = current_path + (goal,)
            return list(goal_path)
        #Agregue los caminos por explorar
        for nodes in new_nodes:
            new_paths += [ current_path + (nodes,)]
        #Los nuevos caminos se agregan como una pila (LIFO)
        new_paths = sort_paths(graph,goal,new_paths)
        new_paths.extend(agenda)
        agenda = new_paths
    print("I couldn't find a path to the goal :(")
    #Retorna una lista vacia si no encuentra un camino al nodo objetivo
    return []

## Ahora implementamos beam search, una variante de BFS
## que acota la cantidad de memoria utilizada para guardar los caminos
## Mantenemos solo k caminos candidatos de tamano n en nuestra agenda en todo momento.
## Los k candidatos deben ser determinados utilizando la
## funcion (valor) de heuristica del grafo, utilizando los valores mas bajos como los mejores
def beam_search(graph, start, goal, beam_width):
    #Se crea la agenda con el primer nodo
    agenda = [(start,)]
    #Ya se alcanzo el objetivo, el nodo de inicio es el objetivo
    if start == goal:
        print('Start node is the same as the goal node')
        return [start]
    #Mientras la agenda no este vacia
    while len(agenda) > 0:
        agenda = agenda[:beam_width]
        #Crea un nuevo camino
        new_paths = []
        while len(agenda) > 0:
            #El primer camino en la agenda es el que hay que extender (FIFO)
            current_path = agenda[0]
            #Actualizar la agenda quitando los caminos explorados
            agenda.remove(current_path)
            #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
            current_node = current_path[-1]
            #Obtenga la lista de nodos conectados al nodo a extender
            new_nodes = graph.get_connected_nodes(current_node)
            #Elimine los nodos repetidos del camino actual
            if len(current_path) > 1:
                new_nodes = [ nodes for nodes in new_nodes if nodes not in current_path]
            #Revise si el objetivo esta en los nodos adyacentes al nodo actual
            if goal in new_nodes:
                goal_path = current_path + (goal,)
                #Regrese la lista con el camino al objetivo (backtracking)
                return list(goal_path)
            #Agregue los caminos por explorar
            for nodes in new_nodes:
                new_paths += [ current_path + (nodes,)]
        #Extienda la agenda con los nuevos caminos
        agenda.extend(new_paths)
        agenda = sort_paths(graph,goal,agenda)
    print("I couldn't find a path to the goal :(")
    #Retorna una lista vacia si no encuentra un camino al nodo objetivo
    return []

## Ahora se implemente busqueda optima, Las anteriores NO utilizan
## las distancias entre los nodos en sus calculos

## Esta funcion toma un grafo y una lista de nombres de nodos y retorna
## la suma de los largos de las aristas a lo largo del camino -- la distancia total del camino.
def path_length(graph, node_names):
    length = 0
    for i in xrange(len(node_names)-1):
        node1 = node_names[i]
        node2 = node_names[i+1]
        length += graph.get_edge(node1, node2).length
    return length

def branch_and_bound(graph, start, goal):
    #Ahora a la agenda se incorpora el costo del camino respectivo
    agenda = [(0, [start])]
    #Mientras la agenda no este vacia
    while len(agenda) > 0:
        #El primer camino en la agenda es el que hay que extender
        current_path = agenda.pop(0)[1]
        #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
        current_node = current_path[-1]
        #Obtenga la lista de nodos conectados al nodo a extender
        new_nodes = graph.get_connected_nodes(current_node)
        #Para cada nuevo nodo
        for node in new_nodes:
            #Descartar los repetidos
            if node not in current_path:
                #Actualizar el camino actual
                new_path = current_path + [node]
                #Nodo actual es el objetivo, anteriormente ya se tiene el camino
                if node == goal:
                    return new_path
                #Actualiza el elemento que contiene el costo y nodos del camino
                updated_path = (path_length(graph, new_path), new_path)
                #Actualizar la agenda
                agenda.append(updated_path)
        #Se ordena la agenda de acuerdo a los costos del camino
        agenda = sorted(agenda, key=lambda tup: tup[0])
    return []


def a_star(graph, start, goal):
    raise NotImplementedError


## Es util determinar si un grafo tiene una heuristica admisible y consistente
## puede dar ejemplos de grafos con heuristica admisible pero no consistente
## consistente pero no admisible?

def is_admissible(graph, goal):
    raise NotImplementedError

def is_consistent(graph, goal):
    raise NotImplementedError

HOW_MANY_HOURS_THIS_PSET_TOOK = '12'
WHAT_I_FOUND_INTERESTING = 'A*'
WHAT_I_FOUND_BORING = 'nada'
