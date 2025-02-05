
from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
import networkx as nx
from py_modules.feature_engineering import get_ProNE_emb, get_node_centrality_nx, get_ones_attrs
cnp.import_array()  # This initializes the numpy C-API




cdef class py_Graph:
    cdef shared_ptr[Graph] inner_graph

    def __cinit__(self,*arg):
        self.inner_graph = shared_ptr[Graph](new Graph())
        cdef int _num_nodes
        cdef int _num_edges
        cdef int[:] edges_from
        cdef int[:] edges_to
        if len(arg)==0:
            deref(self.inner_graph).num_edges=0
            deref(self.inner_graph).num_nodes=0
        elif len(arg)==4:
            _num_nodes=arg[0]
            _num_edges=arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            self.reshape_Graph(_num_nodes,  _num_edges,  edges_from,  edges_to)
        elif len(arg) == 5:
            _num_nodes = arg[0]
            _num_edges = arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            node_attrs = arg[4]  # Assuming it's already a numpy array
            self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to, node_attrs)
        else:
            print('Error: py_Graph class was not successfully initialized. Expected 0, 4, or 5 arguments.')


    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def edge_list(self):
        return deref(self.inner_graph).edge_list

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, double[:, :] node_attrs=None):
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        cdef int i
        cdef vector[vector[double]] c_node_attributes

        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]

        if node_attrs is not None:
            for i in range(node_attrs.shape[0]):
                c_node_attributes.push_back([])
                for j in range(node_attrs.shape[1]):
                    c_node_attributes[i].push_back(node_attrs[i, j])

            self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes, _num_edges, &cint_edges_from[0], &cint_edges_to[0], c_node_attributes))

        else:
            self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0]))
        free(cint_edges_from)
        free(cint_edges_to)

    def reshape(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
        self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to)

    @property
    def node_attributes(self):
        cdef int i, j
        cdef int num_nodes = deref(self.inner_graph).num_nodes
        # Assuming each node has a fixed number of attributes, let's call it num_attrs
        cdef int num_attrs = len(deref(self.inner_graph).node_attributes[0]) if num_nodes > 0 else 0
        node_attrs_np = np.array(deref(self.inner_graph).node_attributes, dtype=np.float64)

        '''
        if(num_attrs == 0):
            return np.empty((0, 0), dtype=np.float64)
        # Create an empty numpy array
        cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.empty((num_nodes, num_attrs), dtype=np.float64)

        # Populate the numpy array
        for i in range(num_nodes):
            for j in range(num_attrs):
                arr[i, j] = deref(self.inner_graph).node_attributes[i][j]
        '''
        return node_attrs_np

    def set_node_attributes(self, double[:, :] node_attrs):
        cdef int i, j
        cdef int num_nodes = deref(self.inner_graph).num_nodes
        cdef int num_attrs = node_attrs.shape[1] if num_nodes > 0 else 0
        cdef vector[vector[double]] c_node_attributes

        for i in range(num_nodes):
            c_node_attributes.push_back([])
            for j in range(num_attrs):
                c_node_attributes[i].push_back(node_attrs[i, j])

        deref(self.inner_graph).node_attributes = c_node_attributes
        
    def set_procedural_attributes(self, procedural_attrs, procedural_attrs_args, covered=None):

        embeddings = self.procedural_node_attributes(covered, procedural_attrs, procedural_attrs_args)
        self.set_node_attributes(embeddings)



    def procedural_node_attributes(self, covered=None, procedural_attrs=[], procedural_attrs_args={}, return_names=False):
        #attributes=['prone', 'degree', 'closeness', 'eigenvector', 'pagerank'],
        #normalize = True, prone_dim=32, return_names=False):

        G_nx = self.to_networkx_graph(covered)

        all_embs = []

        attributes = set(stat.lower() for stat in procedural_attrs)
        prone_emb = None
        if 'prone' in attributes:
            prone_args = procedural_attrs_args.get('prone', {})
            prone_emb, node_ids = get_ProNE_emb(G_nx, **prone_args)
            all_embs.append(prone_emb)
            attributes.remove('prone')

        ones_np = None
        if 'ones' in attributes:
            ones_args = procedural_attrs_args.get('ones', {'num_features': 1})
            ones_np = get_ones_attrs(G_nx, **ones_args)
            attributes.remove('ones')
            all_embs.append(ones_np)

        attributes = list(attributes)
        cent_emb, cent_list_of_features = get_node_centrality_nx(G_nx, attributes, procedural_attrs_args)
        if len(cent_emb) > 0:
            all_embs.append(cent_emb)


        # this is the final feature set. 
        cent_emb = np.concatenate(all_embs, axis=1)
        


        cent_emb = cent_emb.astype(np.double)

        if return_names:
            # create a list of names for attributes cent_names for centrality and prone_{i} for prone embeddings
            emb_names = cent_list_of_features
            if prone_emb is not None:
                prone_dim = prone_emb.shape[1]
                prone_names = [f'prone_{i}' for i in range(prone_dim)]
                emb_names += prone_names

            return cent_emb, emb_names

        else:

            return cent_emb


    def to_networkx_graph(self, covered=None):
        G = nx.Graph()
        covered = set(covered) if covered is not None else set()
        
        # Add nodes
        for i in range(self.num_nodes):
            G.add_node(i)
        
        # Add edges
        for edge in self.edge_list:
            if edge[0] not in covered and edge[1] not in covered:
                G.add_edge(edge[0], edge[1])
        
        return G



cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet())
    # def __dealloc__(self):
    #     if self.inner_gset != NULL:
    #         self.inner_gset.reset()
    #         gc.collect()
    def InsertGraph(self,int gid,py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
        #self.InsertGraph(gid,graph.inner_graph)

        # deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
         #self.Inner_InsertGraph(gid,graph.inner_graph)

    def Sample(self, return_id=False):
        result = deref(self.inner_gset).Sample(return_id)  # We pass the return_id directly
        temp_innerGraph = deref(result.first)  # Extracting Graph from the pair
        gid = result.second  # Extracting gid from the pair
        if return_id:
            return self.G2P(temp_innerGraph), gid
        else:
            return self.G2P(temp_innerGraph) 

    def Get(self,int gid):
        temp_innerGraph=deref(deref(self.inner_gset).Get(gid))   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Clear(self):
        deref(self.inner_gset).Clear()

    cdef G2P(self,Graph graph):
        num_nodes = graph.num_nodes     #得到Graph对象的节点个数
        num_edges = graph.num_edges    #得到Graph对象的连边个数
        edge_list = graph.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int32)
        cint_edges_to = np.zeros([num_edges],dtype=np.int32)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second


        #cdef int num_attrs = len(graph.node_attributes[0]) if num_nodes > 0 else 0
        #if(num_attrs == 0):
        #    return np.empty((0, 0), dtype=np.float64)

        # Converting graph.node_attributes to a numpy array
        node_attrs_mv = np.array(graph.node_attributes, dtype=np.float64)



        return py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to, node_attrs_mv)


        #return py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to)


