'''
file:graph.pxd 类graph的定义文件对应graph.h
'''
#Cython已经编译了C++的std模板库，位置在~/Cython/Includes/lincpp/
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.pair cimport pair
cdef extern from "./src/lib/graph.h":
    ctypedef vector[double] DoubleVector
    ctypedef vector[DoubleVector] DoubleVectorVector
    cdef cppclass Graph:
        Graph()except+
        Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,  DoubleVectorVector _node_attributes)except+
        Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to) except+
        int num_nodes
        int num_edges
        vector[vector[double]] node_attributes
        vector[vector[int]] adj_list
        vector[pair[int,int]] edge_list

cdef extern from "./src/lib/graph.h":
    cdef cppclass GSet:
        GSet()except+
        void InsertGraph(int gid, shared_ptr[Graph] graph)except+
        pair[shared_ptr[Graph], int] Sample(bool return_id) except+
        shared_ptr[Graph] Get(int gid)except+
        void Clear()except+
        map[int, shared_ptr[Graph]] graph_pool



