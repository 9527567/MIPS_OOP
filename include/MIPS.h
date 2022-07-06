#ifndef MIPS_H_INCLUDED
#define MIPS_H_INCLUDED
#include <stdlib.h>
#include <iostream> // cin, cout
#include <fstream> // fscanf, fopen, ofstream
#include <Eigen/Dense>

using namespace Eigen;

#include "header.h"
//Define a base class MIPS with the purpose of solving MIPS problems
//Other algorithms (CEOs, LSH) will inherit this class and extend it
//for their own needs
class MIPS{

protected:
    // Input
    int PARAM_DATA_N; // Number of points (rows) of X
    int PARAM_QUERY_Q; // Number of rows (queries) of Q
    int PARAM_DATA_D; // Number of dimensions

    // Internal params
    bool PARAM_INTERNAL_SAVE_OUTPUT; // save the results

    // MIPS
    int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
    int PARAM_MIPS_TOP_B; // number of points to compute dot products

    // Input
    //Matrix<float, Dynamic, Dynamic, ColMajor> MATRIX_X; // matrix X
    MatrixXf MATRIX_X;
    //extern Matrix<float, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q
    MatrixXf MATRIX_Q;


public:

    MIPS(int N, int Q, int D, int K, int B, bool save);

    void read_matrix_x(const char *path);  //read the file of matrix X MATRIX_X

    void read_matrix_q(const char *path);  //read the file of matrix Q MATRIX_Q

    // compute distance and extract top-K
    void extract_TopK_MIPS(const Ref<VectorXf>&, const Ref<VectorXi>&, int, Ref<VectorXi>);
};








#endif // MIPS_H_INCLUDED
