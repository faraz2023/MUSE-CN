
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph
from libcpp.string cimport string

cdef extern from "./src/lib/mvc_env.h":
    cdef cppclass MvcEnv:
        MvcEnv(double _norm, string _reward_function, int k_dcndp)
        void s0(shared_ptr[Graph] _g)except+
        double step(int a)except+
        void stepWithoutReward(int a)except+
        int randomAction()except+
        int betweenAction()except+
        bool isTerminal()except+
        # double getReward(double oldCcNum)except+
        double getReward()except+
        double getCurrentUnnormalizedScore()except+
        double getMaxConnectedNodesNum()except+
        double getRemainingCNDScore()except+
        double norm
        string reward_function
        int k_dcndp
        double CcNum
        shared_ptr[Graph] graph
        vector[vector[int]]  state_seq
        vector[int] act_seq
        vector[int] action_list
        vector[double] reward_seq
        vector[double] sum_rewards
        vector[double] normalized_returns
        int numCoveredEdges
        set[int] covered_set
        vector[int] avail_list

# cdef class py_MvcEnv:
#     cdef shared_ptr[MvcEnv] inner_MvcEnv
#     cdef shared_ptr[Graph] inner_Graph
#     cdef object original_graph_nx