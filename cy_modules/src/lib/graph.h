#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, const std::vector<std::vector<double>>& _node_attributes);
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to);
    ~Graph();
    int num_nodes;
    int num_edges;
    std::vector< std::vector< int > > adj_list;
    std::vector< std::pair<int, int> > edge_list;
    std::vector<std::vector<double>> node_attributes;
    double getTwoRankNeighborsRatio(std::vector<int> covered);

};

class GSet
{
public:
    GSet();
    ~GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);
    std::pair<std::shared_ptr<Graph>, int> Sample(bool return_id);
    std::shared_ptr<Graph> Get(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;
};

extern GSet GSetTrain;
extern GSet GSetTest;

#endif