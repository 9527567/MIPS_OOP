#include <iostream>
#include "CEOs.h"
#include "CLI11.hpp"
#include "header.h"
#include <string>

using namespace std;

/**
Parameters:
- -N : number of rows of matrix X
- -Q : number of rows of matrix Q
- -D : number of dimensions
- -FX : filename contains the matrix X of format N x D
- -FQ : filename contains the matrix Y of format Q x D
- -M: method and its paramenters : name of method
- -K: TopK largest entries from Xq
- -B: number of points to compute dot products defalut:

Notes:
- The format of dataset should be the matrix format N x D
**/


void args_output(){

    std::cout << "number of rows of matrix X: " << PARAM_DATA_N << std::endl;
    std::cout << "number of rows of matrix Q: " << PARAM_QUERY_Q << std::endl;
    std::cout << "number of dimensions: " <<  PARAM_DATA_D << std::endl;

    std::cout << "file path: " <<  X_FILE_PATH << std::endl;
}





int main(int argc, char **argv)
{
    srand(time(NULL)); // should only be called once for random generator
    CLI::App app{"App description"};

    //setting of base MIPS
    app.add_option<int,unsigned int>("-N", PARAM_DATA_N);
    app.add_option<int,unsigned int>("-Q", PARAM_QUERY_Q);
    app.add_option<int,unsigned int>("-D", PARAM_DATA_D);
    app.add_option<int,unsigned int>("-K", PARAM_MIPS_TOP_K);
    app.add_option<int,unsigned int>("-B", PARAM_MIPS_TOP_B);
    app.add_option<int,unsigned int>("-S", PARAM_MIPS_NUM_SAMPLES);

    string fx = "";
    string fq = "";

    app.add_option("--fx", fx, "X_FILE_PATH");
    app.add_option("--fq", fq, "Q_FILE_PATH");
    //select one method
    int m=0;
    app.add_option<int,unsigned int>("-M", m);
    //CEOs args
    int PARAM_CEOs_D_UP = 1024; //default number of random projections
    app.add_option<int,unsigned int>("--up", PARAM_CEOs_D_UP);
    int PARAM_CEOs_S0 = 5;
    app.add_option<int,unsigned int>("--s0", PARAM_CEOs_S0);
    //data file path



    CLI11_PARSE(app, argc, argv); //phrase args

    X_FILE_PATH = fx.data(); //convert string to char *
    Q_FILE_PATH = fq.data(); //convert string to char *
    args_output();   //print args

    //construct a base CEOs
    coCEOs ceos(PARAM_DATA_N, PARAM_QUERY_Q, PARAM_DATA_D,
                PARAM_MIPS_TOP_K, PARAM_MIPS_TOP_B, PARAM_INTERNAL_SAVE_OUTPUT,
                PARAM_CEOs_D_UP,PARAM_MIPS_NUM_SAMPLES, PARAM_CEOs_S0);


    //read data
    ceos.read_matrix_x(X_FILE_PATH);
    ceos.read_matrix_q(Q_FILE_PATH);
    cout << "read finished!!!!!!" << endl;
    ceos.build_Index();
    ceos.find_TopK();
    //cout << "Hello world!" << endl;
    return 0;
}
