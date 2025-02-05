#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"
#include "disjoint_set.h"

class MvcEnv
{
public:
    MvcEnv(double _norm, std::string _reward_function, int _k_dcndp);

    ~MvcEnv();

    void s0(std::shared_ptr<Graph> _g);

    double step(int a);

    void stepWithoutReward(int a);

    std::vector<double> Betweenness(std::vector<std::vector<int>> adj_list);

    int randomAction();

    int betweenAction();

    bool isTerminal();

    //    double getReward(double oldCcNum);
    double getReward();
    double getCurrentUnnormalizedScore();

    double getMaxConnectedNodesNum();

    double getRemainingCNDScore();
    double getTotalKhopConnectivity(int k);
    std::unordered_set<int> getKhopNeighbors(int node, int k);

    double CcNum;

    void printGraph();

    double norm;
    std::string reward_function;
    int k_dcndp;

    std::shared_ptr<Graph> graph;

    std::vector<std::vector<int>> state_seq;

    std::vector<int> act_seq, action_list;

    std::vector<double> reward_seq, sum_rewards;

    std::vector<double> normalized_returns;

    int numCoveredEdges;

    std::set<int> covered_set;

    std::vector<int> avail_list;

    std::vector<int> node_degrees;

    int total_degrees;
};

#endif