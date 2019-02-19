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
            #Regrese la lista con el camino al objetivo (backtracking)
            return list(goal_path)
        #Agregue los caminos por explorar
        for nodes in new_nodes:
            new_paths += [ current_path + (nodes,)]
        #Extienda la agenda con los nuevos caminos (FIFO)
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
        new_paths.extend(agenda)
        agenda = new_paths
    print("I couldn't find a path to the goal :(")
    #Retorna una lista vacia si no encuentra un camino al nodo objetivo
    return []

## Ahora agregue heuristica a su busqueda
## Hill-climbing puede verse como un tipo de busqueda a profundidad primero
## La busqueda debe ser hacia los valores mas bajos que indica la heuristica

#Retorna los nuevos caminos ordenados ascendentemente de acuerdo a
#la heuristica
#Entrada:
#    graph: el grafo
#    goal: nombre del nodo objetivo
#    new_paths: los caminos por ordenar
#Salida:
#    sorted_paths: lista con los caminos ordenados
def sort_paths(graph,goal,new_paths):
    heuristic_to_goal_list = []
    sorted_paths = []
    for path in new_paths:
        #Retorna el valor de la heuristica del nodo actual al objetivo
        heuristic_to_goal = graph.get_heuristic(path[-1],goal)
        heuristic_to_goal_list.append([path,heuristic_to_goal])
    #Se ordenan los caminos de acuerdo a la heuristica
    heuristic_to_goal_list = sorted(heuristic_to_goal_list, key=lambda x: x[1], reverse=False)
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

#Entrada:
#    graph: el grafo
#    start: nombre del nodo de inicio
#    goal: nombre del nodo objetivo
#Salida:
#    lista: goal_path (lista con el camino al nodo objetivo) o vacia
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

#Entrada:
#    graph: el grafo
#    node_names: el par de nodos para calcular la distancia entre
#Salida:
#    length: el "largo" o costo del camino entre el par de nodos
def path_length(graph, node_names):
    length = 0
    for i in xrange(len(node_names)-1):
        length += graph.get_edge(node_names[i], node_names[i+1]).length
    return length

#Entrada:
#    graph: el grafo
#    start: nombre del nodo de inicio
#    goal: nombre del nodo objetivo
#Salida:
#    lista: goal_path (lista con el camino al nodo objetivo) o vacia
def branch_and_bound(graph, start, goal):
    #Se crea la agenda con el primer nodo
    agenda = [(start,)];
    #Se crea la referencia al camino objetivo
    goal_path = []
    #Ya se alcanzo el objetivo, el nodo de inicio es el objetivo
    if start == goal:
        return [start]
    #Mientras la agenda no este vacia y no este en el camino optimo
    while len(agenda) > 0:
        #Crea un nuevo camino
        new_paths = []
        #El primer camino en la agenda es el que hay que extender
        current_path = agenda[0]
        #Actualizar la agenda quitando los caminos explorados
        agenda.remove(agenda[0])
        #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
        current_node = current_path[-1]
        #Obtenga la lista de nodos conectados al nodo a extender
        new_nodes = graph.get_connected_nodes(current_node)
        #Elimine los nodos repetidos del camino actual
        if len(current_path) > 1:
            new_nodes = [ node for node in new_nodes if node not in current_path]
        #Descarte las opciones que no producen una solucion mas optima que la actual
        if goal in new_nodes:
            #Caso cuando hay no se ha encontrado aun un camino al objetivo
            if len(goal_path) == 0:
                goal_path = current_path + (goal,)
            #Caso donde se deben chequear caminos mas optimos
            if len(goal_path) != 0:
                new_goal_path = current_path + (goal,)
                if path_length(graph,new_goal_path) <= path_length(graph,goal_path):
                    goal_path = new_goal_path
        #Agregue los caminos por explorar
        for nodes in new_nodes:
            new_paths += [ current_path + (nodes,)]
        #Extienda la agenda con los nuevos caminos (LIFO)
        new_paths.extend(agenda)
        agenda = new_paths
    if len(goal_path) != 0:
        return list(goal_path)
    else: return []


#Retorna los nuevos caminos ordenados ascendentemente de acuerdo al
#costo estimado = la heuristica + costo del camino
#Entrada:
#    graph: el grafo
#    goal: nombre del nodo objetivo
#    new_paths: los caminos por ordenar
#Salida:
#    sorted_paths: lista con los caminos ordenados
def sort_paths_a_star(graph,goal,new_paths):
    estimated_cost_to_goal_list = []
    sorted_paths = []
    for path in new_paths:
        #Retorna el valor de la heuristica del nodo actual al objetivo
        heuristic_to_goal = graph.get_heuristic(path[-1],goal)
        #Calcula el costo estimado al objetivo
        estimated_cost=path_length(graph, path)+heuristic_to_goal
        estimated_cost_to_goal_list.append([path,estimated_cost])
    #Se ordenan los caminos de acuerdo a la heuristica
    estimated_cost_to_goal_list = sorted(estimated_cost_to_goal_list, key=lambda x: x[1], reverse=False)
    for paths in estimated_cost_to_goal_list:
        #Se recuperan solo los caminos de la lista ordenada de heuristicas
        sorted_paths.append(paths[0])
    return sorted_paths

def a_star(graph, start, goal):
    #Se crea la agenda con el primer nodo
    agenda = [(start,)];
    #Se crea la referencia al camino objetivo
    goal_path = []
    #Ya se alcanzo el objetivo, el nodo de inicio es el objetivo
    if start == goal:
        return [start]
    #Mientras la agenda no este vacia y no este en el camino optimo
    while len(agenda) > 0 and not goal_path:
        #Crea un nuevo camino
        new_paths = []
        #El primer camino en la agenda es el que hay que extender
        current_path = agenda[0]
        #Actualizar la agenda quitando los caminos explorados
        agenda.remove(agenda[0])
        #Obtenga el nodo a extender, este es el ultimo nodo del camino actual
        current_node = current_path[-1]
        #Obtenga la lista de nodos conectados al nodo a extender
        new_nodes = graph.get_connected_nodes(current_node)
        #Elimine los nodos repetidos del camino actual
        if len(current_path) > 1:
            new_nodes = [ node for node in new_nodes if node not in current_path]
        #Revise si el objetivo esta en los nodos adyacentes al nodo actual,
        #se debe asegurar que sea el camino mas optimo
        if goal in new_nodes:
            goal_path = current_path + (goal,)
        #Agregue los caminos por explorar
        for nodes in new_nodes:
            new_paths += [ current_path + (nodes,)]
        new_paths.extend(agenda)
        #Actualice la agenda
        agenda = new_paths
        #Ordene la agenda de acuerdo al costo estimado
        agenda = sort_paths_a_star(graph,goal,agenda)
    #Si el camino al objetivo se quita de la agenda, se asegura que es
    #el optimo
    if len(goal_path) != 0:
        return list(goal_path)
    else: return []

## Es util determinar si un grafo tiene una heuristica admisible y consistente
## puede dar ejemplos de grafos con heuristica admisible pero no consistente
## consistente pero no admisible?

#Una heuristica admisible es aquella que NUNCA sobreestima
#el costo del camino al objetivo
def is_admissible(graph, goal):
    admissible = True
    #Encuentre el camino optimo para todos los nodos a partir de un
    #objetivo
    for node in graph.nodes:
        path_to_goal = a_star(graph,node,goal)
        if len(path_to_goal) > 0:
            distance_to_goal = path_length(graph,path_to_goal)
            #Si la heuristica al objetivo es mayor a la distancia al
            #objetivo con el camino optimo entonces esta no admisible
            if graph.get_heuristic(node,goal) > distance_to_goal:
                admissible = False
    return admissible

#Una heuristica consistente es aquella que cumple con la desigualdad
#triangular, todo heuristica consistente es admisible pero no
#necesariamente al reves
def is_consistent(graph, goal):
    consistent = True
    #Para todas las aristas del grafo verifique la desigualdad triangular
    for edge in graph.edges:
        if  edge.length < abs(graph.get_heuristic(edge.node2,goal) - graph.get_heuristic(edge.node1,goal)):
            consistent = False
    return consistent

HOW_MANY_HOURS_THIS_PSET_TOOK = 'aprox. 12'
WHAT_I_FOUND_INTERESTING = 'A*, al ser el algoritmo mas ampliamente conocido de best first search'
WHAT_I_FOUND_BORING = 'Ningun problema, todos fueron interesantes'
