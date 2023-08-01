#ifndef TP2_MET_POTENCIA_H
#define TP2_MET_POTENCIA_H

#endif //TP2_MET_POTENCIA_H

#include "eigen-3.4.0/Eigen/Dense"
#include "math.h"
#include "tuple"
#include "vector"
#include "iostream"
#include "fstream"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

pair<double, VectorXd> power_iteration(MatrixXd& A, int niter, float eps);

bool criterio1(VectorXd& v, VectorXd& v0, float eps);

bool criterio2(double a, double a0, float eps);

void calcular_autovalores(MatrixXd& A, int num, int niter, float eps);

// tuple<vector<double>, vector<VectorXd>> calcular_autovalores(MatrixXd A, int num, int niter, float eps);

//matriz mul(matriz A, matriz B);
//
//float norma(matriz A);
//
//matriz div_escalar(matriz A, float e);