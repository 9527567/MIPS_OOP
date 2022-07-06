#ifndef CEOS_H_INCLUDED
#define CEOS_H_INCLUDED
#include <iostream>

#include "MIPS.h"
#include "fht.h"
#include "Utilities.h"
#include "header.h"

#include <Eigen/Dense>
using namespace Eigen;

//Define the class of the base 1CEOs, it is inherited from MIPS.
class OneCEOs:public MIPS{

protected:

    // CEOs setting parameters
    int PARAM_CEOs_D_UP;
    int PARAM_CEOs_NUM_ROTATIONS; // Use Gauss or HD3
    int PARAM_INTERNAL_LOG2_CEOs_D_UP;
    bool PARAM_INTERNAL_NOT_STORE_MATRIX_X;

    MatrixXf PROJECTED_X;
    MatrixXi HD3;                //Matrix generated by HD3
    MatrixXf MATRIX_G;           //Matrix generated by Gauss

    // CEOs
    MatrixXi MATRIX_CEOs;

    void rotateData();              //a function used in build_Index function
    void HD3Generator(int p_iSize);
    void gaussGenerator(int p_iNumRows, int p_iNumCols);

public:

    OneCEOs(int N, int Q, int D,int K, int B, bool save, int U):MIPS(N, Q, D, K, B, save)
    {
        this->PARAM_CEOs_NUM_ROTATIONS = 3; //default setting
        this->PARAM_INTERNAL_NOT_STORE_MATRIX_X = false;
        this->PARAM_CEOs_D_UP = U;
    };

    //each CEOs has their own build_Index and find_TopK functions.
    virtual void build_Index();
    virtual void find_TopK();

};


//Define the class of the 2CEOs, it is inherited from 1CEOs.
class TwoCEOs:public OneCEOs{

public:
    TwoCEOs(int N, int Q, int D,int K, int B, bool save, int U):
    OneCEOs(N, Q, D, K, B, save, U){};

    void build_Index();
    void find_TopK();
};
#endif // CEOS_H_INCLUDED



//Define the class of the sCEOs, it is inherited from OneCEOs.
class sCEOsEst:public OneCEOs{


protected:

    int PARAM_CEOs_S0;
    // extract topB&topK, it's overwrited in coCEOs.
    virtual void extract_TopB_TopK_Histogram(const Ref<VectorXf>&, const Ref<VectorXf>&, int, int, Ref<VectorXi>);
    // extract topK min&max index of query
    void extract_TopK_MinMax_Idx(const Ref<VectorXf>&, int, Ref<VectorXi>, Ref<VectorXi>);

public:
    sCEOsEst(int N, int Q, int D,int K, int B , bool save, int U, int S):
    OneCEOs(N, Q, D, K, B, save, U){ PARAM_CEOs_S0 = S;};

    virtual void build_Index();
    virtual void find_TopK();
};




//Define the class of the sCEOs_TA, it it inherited from sCEOsEst
/**
Struct of comparing matrix elements
**/
struct compare_PROJECTED_X
{
    int dim;
    MatrixXf P_X;

    compare_PROJECTED_X(int d, MatrixXf PROJECTED_X){
        dim = d;
        P_X = PROJECTED_X;
    }
    bool operator() (int i, int j) { return (P_X(i, dim) > P_X(j, dim));}
};


class sCEOsTA : public sCEOsEst{

protected:

    MatrixXi CEOs_TA_SORTED_IDX;

public:

    sCEOsTA(int N, int Q, int D,int K, int B, bool save, int U, int S):
    sCEOsEst(N, Q, D, K, B, save, U, S){};

    void build_Index();
    void find_TopK();

};


//Define the class of the coCEOs, it it inherited from OneCEOs
class coCEOs : public sCEOsEst{

protected:

    int PARAM_CEOs_N_DOWN;
    int PARAM_MIPS_NUM_SAMPLES; // Budgeted number of points to compute dot products
    vector<IFPair> coCEOs_MAX_IDX; // co-CEOs
    vector<IFPair> coCEOs_MIN_IDX; // co-CEOs

    //Overwrite this funtion
    void extract_TopB_TopK_Histogram(const unordered_map<int, float> &, const Ref<VectorXf> &,
                                 int, int , Ref<VectorXi>);


    void Vector_TopK();
    void Map_TopK();




public:
    coCEOs(int N, int Q, int D,int K, int B, bool save, int U, int samples, int S):
    sCEOsEst(N, Q, D, K, B, save, U, S){
        PARAM_MIPS_NUM_SAMPLES = samples;
        PARAM_CEOs_N_DOWN = ceil(PARAM_MIPS_NUM_SAMPLES / (2 * PARAM_CEOs_S0));
    };

    void build_Index();
    void find_TopK();

};
