from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libcpp cimport bool
import  graph
import  numpy as np
# import gc


cdef class py_ReplaySample:
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    def __cinit__(self,int batch_size):
        self.inner_ReplaySample = shared_ptr[ReplaySample](new ReplaySample(batch_size))
    # def __dealloc__(self):
    #     if self.inner_ReplaySample != NULL:
    #         self.inner_ReplaySample.reset()
    #         gc.collect()
    @property
    def g_list(self):
        result = []
        for graphPtr in deref(self.inner_ReplaySample).g_list:
            result.append(self.G2P(deref(graphPtr)))
        return  result
    @property
    def list_st(self):
        return deref(self.inner_ReplaySample).list_st
    @property
    def list_s_primes(self):
        return deref(self.inner_ReplaySample).list_s_primes
    @property
    def list_at(self):
        return deref(self.inner_ReplaySample).list_at
    @property
    def list_rt(self):
        return deref(self.inner_ReplaySample).list_rt
    @property
    def list_term(self):
        return deref(self.inner_ReplaySample).list_term
    @property
    def list_normalized_rt(self):
        return deref(self.inner_ReplaySample).list_normalized_rt

    cdef G2P(self,Graph graph1):
        num_nodes = graph1.num_nodes     
        num_edges = graph1.num_edges    
        edge_list = graph1.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int32)
        cint_edges_to = np.zeros([num_edges],dtype=np.int32)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second

        node_attrs_mv = np.array(graph1.node_attributes, dtype=np.float64)
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to, node_attrs_mv)


cdef class py_NStepReplayMem:
    cdef shared_ptr[NStepReplayMem] inner_NStepReplayMem
    cdef shared_ptr[Graph] inner_Graph
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    def __cinit__(self,int memory_size):
        self.inner_NStepReplayMem = shared_ptr[NStepReplayMem](new NStepReplayMem(memory_size))
    # def __dealloc__(self):
    #     if self.inner_NStepReplayMem != NULL:
    #         self.inner_NStepReplayMem.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    #     if self.inner_ReplaySample != NULL:
    #         self.inner_ReplaySample.reset()
    #         gc.collect()


    def Add(self,mvcenv,int nstep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        # g = self.GenNetwork(mvcenv.graph)
        g = mvcenv.graph
        deref(self.inner_Graph).num_nodes= g.num_nodes
        deref(self.inner_Graph).num_edges=g.num_edges
        deref(self.inner_Graph).edge_list=g.edge_list
        deref(self.inner_Graph).adj_list=g.adj_list
        deref(self.inner_Graph).node_attributes = g.node_attributes

        mvc_reward = mvcenv.reward_function.encode()
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(mvcenv.norm, mvc_reward, mvcenv.k_dcndp))
        deref(self.inner_MvcEnv).norm = mvcenv.norm
        deref(self.inner_MvcEnv).graph = self.inner_Graph
        deref(self.inner_MvcEnv).state_seq = mvcenv.state_seq
        deref(self.inner_MvcEnv).act_seq = mvcenv.act_seq
        deref(self.inner_MvcEnv).action_list = mvcenv.action_list
        deref(self.inner_MvcEnv).reward_seq = mvcenv.reward_seq
        deref(self.inner_MvcEnv).sum_rewards = mvcenv.sum_rewards
        deref(self.inner_MvcEnv).normalized_returns = mvcenv.normalized_returns
        deref(self.inner_MvcEnv).numCoveredEdges = mvcenv.numCoveredEdges
        deref(self.inner_MvcEnv).covered_set = mvcenv.covered_set
        deref(self.inner_MvcEnv).avail_list = mvcenv.avail_list
        deref(self.inner_NStepReplayMem).Add(self.inner_MvcEnv,nstep)

    def Sampling(self,int batch_size):
        # self.inner_ReplaySample = shared_ptr[ReplaySample](new ReplaySample(batch_size))
        self.inner_ReplaySample =  deref(self.inner_NStepReplayMem).Sampling(batch_size)
        result = py_ReplaySample(batch_size)
        result.inner_ReplaySample = self.inner_ReplaySample
        return result

    @property
    def graphs(self):
        result = []
        for graphPtr in deref(self.inner_NStepReplayMem).graphs:
            result.append(self.G2P(deref(graphPtr)))
        return  result
    @property
    def actions(self):
        return deref(self.inner_NStepReplayMem).actions
    @property
    def rewards(self):
        return deref(self.inner_NStepReplayMem).rewards
    @property
    def states(self):
        return deref(self.inner_NStepReplayMem).states
    @property
    def s_primes(self):
        return deref(self.inner_NStepReplayMem).s_primes
    @property
    def terminals(self):
        return deref(self.inner_NStepReplayMem).terminals
    @property
    def current(self):
        return deref(self.inner_NStepReplayMem).current
    @property
    def count(self):
        return deref(self.inner_NStepReplayMem).count
    @property
    def memory_size(self):
        return deref(self.inner_NStepReplayMem).memory_size
    cdef G2P(self,Graph graph1):
        num_nodes = graph1.num_nodes     #得到Graph对象的节点个数
        num_edges = graph1.num_edges    #得到Graph对象的连边个数
        edge_list = graph1.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int32)
        cint_edges_to = np.zeros([num_edges],dtype=np.int32)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
        node_attrs_mv = np.array(graph1.node_attributes, dtype=np.float64)
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to, node_attrs_mv)

    '''
    def GenNetwork(self, g):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)
    '''