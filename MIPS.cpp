#include "MIPS.h"
using namespace std;

//CONSTRUCT
MIPS::MIPS(int N, int Q, int D, int K, int B, bool save)
{

    PARAM_DATA_N = N; // Number of points (rows) of X
    PARAM_QUERY_Q = Q; // Number of rows (queries) of Q
    PARAM_DATA_D = D; // Number of dimensions

    // MIPS
    PARAM_MIPS_TOP_K = K; // TopK largest entries from Xq
    PARAM_MIPS_TOP_B = B; // number of points to compute dot products

    // Internal params
    PARAM_INTERNAL_SAVE_OUTPUT = save; // save the results

    printf("N:%d, Q:%d, D:%d, K:%d, B:%d\n",N,Q,D,K,B);
}

//read the Matrix X from path
void MIPS::read_matrix_x(const char *path){
    FILE *f = fopen(path, "r");
    if (!f)
    {
        printf("Data file does not exist");
        exit(1);
    }

    FVector vecTempX(PARAM_DATA_D * PARAM_DATA_N, 0.0);

    // Each line is a vector of D dimensions
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            fscanf(f, "%f", &vecTempX[n * PARAM_DATA_D + d]);
            //cout << vecTempX[n * PARAM_DATA_D + d] << " ";
        }
        // cout << endl;
    }

    // Matrix_X is col-major
    MATRIX_X = Map<MatrixXf>(vecTempX.data(), PARAM_DATA_D, PARAM_DATA_N);

}


//read the Matrix Q from path
void MIPS::read_matrix_q(const char *path){

    FILE *f = fopen(path, "r+");
    if (!f)
    {
        printf("Data file does not exist");
        exit(1);
    }

    FVector vecTempQ(PARAM_DATA_D * PARAM_QUERY_Q, 0.0);
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            fscanf(f, "%f", &vecTempQ[q * PARAM_DATA_D + d]);
            //cout << vecTempQ[q * D + d] << " ";
        }
        //cout << endl;
    }

    MATRIX_Q = Map<MatrixXf>(vecTempQ.data(), PARAM_DATA_D, PARAM_QUERY_Q);

}


/** \brief Return top K from VectorXi top B for postprocessing step
 *
 * \param
 *
 - VectorXf::vecQuery: vector query
 - VectorXi vectopB: vector of Top B indexes
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top-K point indexes
 *
 */
void MIPS::extract_TopK_MIPS(const Ref<VectorXf> &p_vecQuery, const Ref<VectorXi>& p_vecTopB, int p_iTopK,
                 Ref<VectorXi> p_vecTopK)
{
    // incase we do not have enough candidates
    assert((int)p_vecTopB.size() >= p_iTopK);

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;


    //for (int n = 0; n < (int)p_vecTopB.size(); ++n)
    for (const auto& iPointIdx: p_vecTopB)
    {
        // Get point Idx
        //int iPointIdx = p_vecTopB(n);
        float fValue = 0.0;

        /**
        // This code is used for CEOs_TA; otherwise, we do not this condition
        if (PARAM_INTERNAL_NOT_STORE_MATRIX_X && PARAM_CEOs_NUM_ROTATIONS)
            // Now: p_vecQuery is the projected query and hence we use Project X of N x Dup
            // It will be slower than the standard case due to col-wise Project_X and D_up > D
            fValue = PROJECTED_X.row(iPointIdx) * p_vecQuery;
        else
        */
        fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }


    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}


