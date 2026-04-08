#ifndef SOP_SOLVER_INTERFACE_H
#define SOP_SOLVER_INTERFACE_H

#include <Eigen/Eigen>
#include <vector>

using std::vector;

int solveSOP(const Eigen::MatrixXi &cost_matrix, vector<int> &path);

#endif