#include <iostream>
#include "met_potencia.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " input_file" << std::endl;
        return 1;
    }
    const char* input_file = argv[1];
//    const char* output_file = argv[2];

    std::ifstream fin(input_file);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open input file " << input_file << std::endl;
        return 1;
    }

    // Read matrix and vector from file
    int nrows, ncols, niter;
    float eps;
    fin >> nrows >> ncols >> niter >> eps;

    MatrixXd A(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> A(i, j);
        }
    }

    fin.close();

    // --- Testing ---
    // power_iteration(A, niter, eps);
    // VectorXd w(nrows);
    // for (int i = 0; i < nrows; i++) {
    //     w(i) = 1;
    // }
    // A = A - 2 * w * w.transpose();
    // return 0;
    // --- Testing ---

    // Perform power_iteration
    // tuple<vector<double>, vector<VectorXd>> res = calcular_autovalores(A, nrows, niter, eps);
    calcular_autovalores(A, nrows, niter, eps);
    // vector<double> autovalores = get<0>(res);
    // vector<VectorXd> autovectores = get<1>(res);

    // // Write result to output file
    // const char* output1 = "autovalores.txt";
    // const char* output2 = "autovectores.txt";

    // std::ofstream fout1(output1);
    // if (!fout1.is_open()) {
    //     std::cerr << "Error: could not open output file " << output1 << std::endl;
    //     return 1;
    // }

    // VectorXd v(nrows);
    // for (int i = 0; i < autovalores.size(); i++) {
    //     v(i) = autovalores[i];
    // }
    // fout1 << v.transpose();
    // fout1.close();

    // std::ofstream fout2(output2);
    // if (!fout2.is_open()) {
    //     std::cerr << "Error: could not open output file " << output2 << std::endl;
    //     return 1;
    // }

    // MatrixXd M(nrows, ncols);
    // for (int j = 0; j < autovectores.size(); j++) {
    //     M.col(j) = autovectores[j];
    // }
    // fout2 << M;
    // fout2.close();

    return 0;
}
