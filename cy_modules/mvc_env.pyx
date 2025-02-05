from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
import networkx as nx
from graph cimport Graph
import gc
from libc.stdlib cimport free
from libcpp.string cimport string
from py_modules.feature_engineering import get_ProNE_emb, get_node_centrality_nx
from scipy.special import softmax


cdef class py_MvcEnv:
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    cdef shared_ptr[Graph] inner_Graph
    cdef object original_graph_nx
    
    def __cinit__(self,double _norm, object _reward_function='cndp', int _k_dcndp=0):
        cdef string reward_function_str_cpp = _reward_function.encode()
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm, reward_function_str_cpp, _k_dcndp))
        self.inner_Graph =shared_ptr[Graph](new Graph())
        self.original_graph_nx = None
    # def __dealloc__(self):
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    def s0(self,_g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).node_attributes = _g.node_attributes
        deref(self.inner_MvcEnv).s0(self.inner_Graph)

        self.original_graph_nx = nx.Graph()
        self.original_graph_nx.add_nodes_from(range(_g.num_nodes))
        edge_list_np = np.array(_g.edge_list)
        self.original_graph_nx.add_edges_from(zip(edge_list_np[:,0], edge_list_np[:,1]))

    @property
    def current_graph_nx(self):
        return self.graph.to_networkx_graph(self.covered_set)

    def imitateAction(self, str imitation_algorithm, bool imitation_deterministic):
        if imitation_algorithm in ['degree', 'pagerank']:
            scores = get_node_centrality_nx(self.current_graph_nx, [imitation_algorithm])[0]
            scores = scores.flatten()
            
            if imitation_deterministic:
                action = np.argmax(scores)
            else:
                raise ValueError(f"Stochastic imitation is not supported yet")
                '''
                avialble_nodes = []
                available_node_scores = []
                for node in range(len(scores)):
                    if node in self.covered_set:
                        pass
                    else:
                        avialble_nodes.append(node)
                        available_node_scores.append(scores[node])

                if(len(available_node_scores) < 5):
                    action = np.argmax(scores)
                else:
                    p_scores = available_node_scores / (np.sum(available_node_scores) + 1e-10)

                    action = np.random.choice(avialble_nodes, p=p_scores)
                '''

            return action
        else:
            raise ValueError(f"Unsupported imitation algorithm: {imitation_algorithm}")

    def step(self,int a):
        return deref(self.inner_MvcEnv).step(a)

    def stepWithoutReward(self,int a):
        deref(self.inner_MvcEnv).stepWithoutReward(a)

    def randomAction(self):
        return deref(self.inner_MvcEnv).randomAction()

    def betweenAction(self):
        return deref(self.inner_MvcEnv).betweenAction()

    def isTerminal(self):
        return deref(self.inner_MvcEnv).isTerminal()

    def getReward(self):
        return deref(self.inner_MvcEnv).getReward()

    def getMaxConnectedNodesNum(self):
        return deref(self.inner_MvcEnv).getMaxConnectedNodesNum()

    def getRemainingCNDScore(self):
        return deref(self.inner_MvcEnv).getRemainingCNDScore()

    def getCurrentUnnormalizedScore(self):
        return deref(self.inner_MvcEnv).getCurrentUnnormalizedScore()

    @property
    def norm(self):
        return deref(self.inner_MvcEnv).norm

    @property
    def reward_function(self):
        return deref(self.inner_MvcEnv).reward_function.decode()

    @property
    def k_dcndp(self):
        return deref(self.inner_MvcEnv).k_dcndp

    @property
    def graph(self):
        # temp_innerGraph=deref(self.inner_Graph)   #得到了Graph 对象
        return self.G2P(deref(self.inner_Graph))

    @property
    def state_seq(self):
        return deref(self.inner_MvcEnv).state_seq

    @property
    def act_seq(self):
        return deref(self.inner_MvcEnv).act_seq

    @property
    def action_list(self):
        return deref(self.inner_MvcEnv).action_list

    @property
    def reward_seq(self):
        return deref(self.inner_MvcEnv).reward_seq

    @property
    def sum_rewards(self):
        return deref(self.inner_MvcEnv).sum_rewards

    @property
    def normalized_returns(self):
        return deref(self.inner_MvcEnv).normalized_returns

    @property
    def numCoveredEdges(self):
        return deref(self.inner_MvcEnv).numCoveredEdges

    @property
    def covered_set(self):
        return deref(self.inner_MvcEnv).covered_set

    @property
    def avail_list(self):
        return deref(self.inner_MvcEnv).avail_list


    cdef G2P(self,Graph graph1):
        num_nodes = graph1.num_nodes     
        num_edges = graph1.num_edges    
        edge_list = graph1.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int32)
        cint_edges_to = np.zeros([num_edges],dtype=np.int32)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
        

        if (num_nodes > 0):
            node_attrs_mv = np.array(graph1.node_attributes, dtype=np.float64)
        
            return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to, node_attrs_mv)
        else:
            return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to)


    # cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
    #     cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int i
    #     for i in range(_num_edges):
    #         cint_edges_from[i] = edges_from[i]
    #     for i in range(_num_edges):
    #         cint_edges_to[i] = edges_to[i]
    #     free(cint_edges_from)
    #     free(cint_edges_to)
    #     return  new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0])