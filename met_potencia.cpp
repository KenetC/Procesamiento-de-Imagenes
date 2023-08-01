#include "met_potencia.h"

pair<double, VectorXd> power_iteration(MatrixXd& A, int niter, float eps) {
    double a = 1;
    VectorXd v = VectorXd::Random(A.rows());
    // vector<double> angulos;
    // for (int i = 0; i < A.rows(); i++) {
    //     v(i) = 1;
    // }
    VectorXd v0 = v;
    VectorXd Bv = v;
    for (int j = 0; j < niter; j++) {
        double a0 = a;
        v0 = v;
        Bv = A * v;
        v = Bv / Bv.norm();
        a = v.transpose() * A * v;
        // criterio de parada
        if (abs(a - a0) < eps) {
            printf("Iteracion: %d\n", j);
            break;
        }
    }
    // a = v.transpose() * A * v;
    pair<double, VectorXd> res;
    res = make_pair(a, v);
    return res;
}

bool criterio1(VectorXd& v, VectorXd& v0, float eps) {
    double cos_ang = v.dot(v0);
    // angulos.emplace_back(cos_ang);
    bool res;
    res = (1 - eps) < fabs(cos_ang) && fabs(cos_ang) <= 1;
    return res;
}

bool criterio2(double a, double a0, float eps) {
    return fabs(a - a0) < eps;
}

void calcular_autovalores(MatrixXd& A, int num, int niter, float eps) {
    // MatrixXd A_copy = A;
    const char* output1 = "autovalores.txt";
    const char* output2 = "autovectores.txt";
    ofstream fout1(output1);
    ofstream fout2(output2);

    // vector<double> eigenvalues(num);
    // vector<VectorXd> eigenvectors(num);
    VectorXd autovalores(num);
    MatrixXd autovectores(num, num);

    for (int i = 0; i < num; i++) {
        pair<double, VectorXd> calculo = power_iteration(A, niter, eps);
        double lam = calculo.first;
        VectorXd vec = calculo.second;
        
        // eigenvalues[i] = lam;
        // eigenvectors[i] = vec;
        autovalores(i) = lam;
        autovectores.col(i) = vec;

        A = A - lam * vec * vec.transpose();
    }

    fout1 << autovalores;
    fout1.close();
    fout2 << autovectores;
    fout2.close();
    // tuple<vector<double>, vector<VectorXd>> res;
    // res = make_tuple(eigenvalues, eigenvectors);
    // return res;
}