import numpy as np
from abc import ABCMeta, abstractmethod

EDGE_NOT_FILLED_ERROR = 'Edge has not received any input from previous node.'

class GraphInconsistencyError(Exception):
    pass

INPUTS_NAME = 'prev_activations'
GRADIENTS_NAME = 'delta'

class Graph:
    """A graph to be used in models to represent the connections between layers

       Note: Technically, the graph could be said to represent two separate directed acyclic graphs - one for the forward model
             computations and one for the backpropagation calculations (called parent-child and child-parent directions respectively).
             When calculating model activations, information can only pass from parent nodes(layers) to children nodes(layers). When
             backpropagating, the information flow restrictions are reversed.

       Parameters:
           'node_connections' - A list of tuples representing node connections. Should be in the format:
                                (parent_node, child_node, (forward_name, backward_name)), where parent_node
                                and child_node are of type Node, and forward_name/backward_name are labels to
                                the edges connecting the two nodes to the intermediate node which the graph creates
                                (of type Edge)
                                           
    """
    def __init__(self, node_connections):
        self.nodes = set()
        self.edges = set()
        self.input_edges = set()
        self.output_edges = set()

        for connection in node_connections:
            parent_node, child_node, (fwd_name, bwd_name) = connection

            edge = Edge(parent_node=parent_node, child_node=child_node, forward_name=fwd_name, backward_name=bwd_name)
            self.edges.add(edge)
            parent_node.child_edges.add(edge)
            child_node.parent_edges.add(edge)

            if parent_node not in self.nodes:
                self.nodes.add(parent_node)
            if child_node not in self.nodes:
                self.nodes.add(child_node)

        self._organize_input_elements()
        self._organize_output_elements()
        self.edges = self.edges.union(*[self.input_edges, self.output_edges])
        self._check_graph_consistency()

    @property
    def outputs(self):
        return [e.give_message(forward=True) for e in self.output_edges]

    def forward(self, inputs_dict, additional_params):
        """ Forward pass of the graph.
            Parameters:
                'inputs_dict'          - Dictionary containing the inputs, with the input edges' names as keys
                'additional_params'    - Aditional parameters to be passed (e.g. train_mode)
        """
        for edge in self.input_edges:
            edge_name = edge.name(forward=False)
            if edge_name in inputs_dict:
                edge.receive_input(inputs_dict[edge_name], forward=True)

        nodes = self.input_nodes
        node_counter = 3
        while len(nodes) > 0:
            for node in nodes:
                node.forward_message(additional_params=additional_params)
                node_counter += len(node.child_edges)

            nodes = self.__get_children(nodes, skip_blocked = True)

    def backward(self, gradients_dict, additional_params, node_func = None):
        """ Backward pass of the graph.
            Parameters:
                'gradients_dict'       - Dictionary containing the output gradients, with the output edges' names as keys
                'additional_params'    - Aditional parameters to be passed (e.g. train_mode)
                'node_func'            - Function to be applied to nodes after gradient computation. Main intent is to allow
                                         the model's optimizer to update the node/layer weights 
        """
        for edge in self.output_edges:
            edge.receive_input(gradients_dict[edge.name(forward=True)], forward=False)

        nodes = self.output_nodes
        while len(nodes) > 0:
            for node in nodes:
                helper_params = additional_params.copy()
                helper_params.update({e.name(forward=True): e.give_message(forward=True) for e in node.parent_edges.union(node.input_edges.values())})

                node.backward_message(helper_params)

                if node_func is not None:
                    node_func(node)

            nodes = self.__get_parents(nodes, skip_blocked = True)

        self.clear_messages()

    def print_graph(self):
        edges = set()
        nodes = self.input_nodes
        while len(nodes) > 0:
           edges = edges.union({e for n in nodes for e in n.child_edges})
           nodes = self.__get_children(nodes)

        for edge in edges:
            print(f'{None if edge.parent_node is None else edge.parent_node._name}-->{None if edge.child_node is None else edge.child_node._name}')
    
    def _organize_input_elements(self):
        """ Creates the graph's input edges
        """
        self.input_nodes = [n for n in self.nodes if len(n.input_edges) > 0]
        self.root_nodes = [n for n in self.nodes if len(n.parent_edges) == 0]

        for node in self.input_nodes:
           for input_name in node.input_edges.keys():
                input_edge = Edge(parent_node=None, child_node=node, backward_name=input_name, forward_name = INPUTS_NAME)
                node.input_edges[input_name] = input_edge
                self.input_edges.add(input_edge)

    def _organize_output_elements(self):
        """ Creates the graph's output edges
        """
        self.output_nodes = [n for n in self.nodes if len(n.child_edges) == 0]
        for node in self.output_nodes:
            for output_name in node.output_edges.keys():
                output_edge = Edge(parent_node=node, child_node=None, backward_name=GRADIENTS_NAME, forward_name=output_name)
                node.output_edges[output_name] = output_edge
                self.output_edges.add(output_edge)

    def _check_graph_consistency(self):
        """ Checks for cycles in the forward graph. The way that the Edge class is constructed means that if
            there are no cycles in the parent-child direction, then there would not be any cycles in the child-parent
            backpropagating phase either 
        """
        for node in self.root_nodes:
            self.__dfs_check_for_cycles(node, parents = set())    

    def clear_messages(self):
        """ Clear "messages" in all the edges of the graph to allow recomputation
        """
        for e in self.edges:
            e.clear_messages()

    def __get_children(self, nodes, skip_blocked = False):
        # skip_blocked would ignore nodes that get input from more than one parent node
        # and not all of their parents have computed their outputs
        if skip_blocked:
            children = [c for n in nodes for c in n.get_children() if c.can_send_forward and not c.frozen]
        else:
            children = [c for n in nodes for c in n.get_children()]

        if len(children) > 0:
            return set(children)
        
        return set()

    def __get_parents(self, nodes, skip_blocked = False):
        # skip_blocked would ignore nodes that get input from more than one child node
        # and not all of their children have computed their outputs
        if skip_blocked:
            parents = [p for n in nodes for p in n.get_parents() if p.can_send_backward and not p.frozen]
        else:
            parents = [p for n in nodes for p in n.get_parents()]

        if len(parents) > 0:
            return set(parents)
        
        return set()

    def __dfs_check_for_cycles(self, node, parents):
        children = node.get_children()
        if len(children) == 0:
            return

        if any([n in parents for n in children]):
                raise GraphInconsistencyError('Graph cannot contain cycles.')

        parents.add(node)
        for child in children:
            self.__dfs_check_for_cycles(child, parents.copy())


class Node(metaclass = ABCMeta):
    """Class representing a layer in the model's graph. Every layer should inherit Node.

       Warning: Every node implicitly handles multiple inputs with the same name by summing them.
       To modify the default behavior, use method set_multiple_inputs_handle(handle). Valid handles:
            - 'sum'
            - 'multiply'
            - 'vstack'
            - 'hstack'
    """
    def __init__(self, input_names = [], output_names = [], name = None):
        self.node_name = name
        self.input_edges = {}
        self.output_edges = {}
        self.frozen = False

        self._inputs_handle = 'sum'

        for name in input_names:
            self.input_edges[name] = None
        for name in output_names:
            self.output_edges[name] = None

        self.parent_edges = set()
        self.child_edges = set()

    @property
    def can_send_forward(self):
        return len([oe for oe in self.parent_edges if not oe.filled(True)]) == 0

    @property
    def can_send_backward(self):
        return len([oe for oe in self.child_edges if not oe.filled(False)]) == 0

    def get_parents(self):
        return set([e.parent_node for e in self.parent_edges if e.parent_node is not None])

    def get_children(self):
        return set([e.child_node for e in self.child_edges if e.child_node is not None])

    def get_inputs(self, forward = True):
        # forward represents whether we are currently moving messages from parents to children
        # (model.forward), or from children to parents (model.backward)
        if forward:
            edges = self.parent_edges.union(set(self.input_edges.values()))
        else:
            edges = self.child_edges.union(set(self.output_edges.values()))

        inputs = {}
        for edge in edges:
            edge_name = edge.name(forward)
            edge_message = edge.give_message(forward)
            if edge_name not in inputs:
                inputs[edge_name] = [edge_message]
            else:
                inputs[edge_name].append(edge_message)

        for key in inputs.keys():
            if len(inputs[key]) == 1:
                inputs[key] = inputs[key][0]
            else:
                inputs[key] = self._handle_multiple_inputs(inputs[key])

        return inputs

    def forward_message(self, additional_params = {}):
        inputs = self.get_inputs(forward = True)
        inputs.update(additional_params)

        message = self._compute_forward_message(inputs)

        for edge in self.child_edges.union(self.output_edges.values()):
            edge.receive_input(message, forward=True)

    def backward_message(self, additional_params = {}):
        inputs = self.get_inputs(forward = False)
        inputs.update(additional_params)

        message = self._compute_backward_message(inputs)

        for edge in self.parent_edges:#.union(self.input_edges.values()):
            edge.receive_input(message, forward=False)

    def set_multiple_inputs_handle(self, handle):
        self._inputs_handle = handle

    def clear_child_edges(self):
        for edge in self.child_edges.union(set(self.output_edges.values())):
            edge.clear_messages()

    def _handle_multiple_inputs(self, inputs):
        if self._inputs_handle == 'sum':
            return sum(inputs)
        elif self._inputs_handle == 'multiply':
            return np.multiply(*inputs)
        elif self._inputs_handle == 'hstack':
            return np.hstack(inputs)
        elif self._inputs_handle == 'vstack':
            return np.vstack(inputs)

    @abstractmethod
    def _compute_forward_message(self, inputs):
        pass

    @abstractmethod
    def _compute_backward_message(self, inputs):
        pass    

class Edge:
    """ Class playing the role of intermediate information storage facility in the connections between nodes.
        Technically, represents an intermediate node in the graph, adding two actual "edges".
    """
    def __init__(self, parent_node, child_node, forward_name = None, backward_name = None):
        self.__parent_node = parent_node
        self.__child_node = child_node
        self.__name_fwd = forward_name
        self.__name_bck = backward_name
        self.__input_fwd = None
        self.__input_bck = None

    @property
    def parent_node(self): return self.__parent_node
    @property
    def child_node(self): return self.__child_node
    
    def filled(self, forward):
        if forward:
            return self.__input_fwd is not None
        return self.__input_bck is not None

    def receive_input(self, input, forward):
        if forward:
            self.__input_fwd = input
        else:
            self.__input_bck = input

    def give_message(self, forward):
        if forward:
            if not self.filled(True):
                raise ValueError(EDGE_NOT_FILLED_ERROR)
            input = self.__input_fwd
        else:
            if not self.filled(False):
                raise ValueError(EDGE_NOT_FILLED_ERROR)
            input = self.__input_bck

        return input

    def clear_messages(self):
        self.__input_fwd = None
        self.__input_bck = None

    def name(self, forward):
        if forward:
            return self.__name_fwd

        return self.__name_bck