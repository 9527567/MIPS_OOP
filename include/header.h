#ifndef HEADER_H_INCLUDED
#define HEADER_H_INCLUDED

#pragma once
//#include "fht.h"

#include <cstdlib> // for abs, atoi
#include <chrono> // for wall clock time
#include <iostream> // cin, cout
#include <cstdio> // printf()

//#include <math.h> // ceil()
//#include <cmath> // cos(), pow()
#include <algorithm>


#include <vector>
#include <utility>
#include <unordered_map>
#include <queue>
#include <unordered_set>

#include <string>
#include <time.h> // for time(0) to generate different random number
#include <random>

using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include <boost/dynamic_bitset.hpp>




#define PI				3.141592653589793238460
#define PRIME           2147483647 // 2^19 - 1 or 2^31 - 1 = 2147483647
#define EPSILON         0.000001

typedef vector<int> IVector;
typedef vector<float> FVector;

/*
Structure of pair
- point index
- dot product
*/
struct IFPair
{
    int m_iIndex;
    double	m_fValue;

	IFPair()
	{
        m_iIndex	= 0;
        m_fValue	= 0.0;
	}

	IFPair(int p_iIndex, float p_fValue)
	{
		m_iIndex	= p_iIndex;
		m_fValue	= p_fValue;
	}

	// Overwrite operation < to get top K largest entries
    bool operator<(const IFPair& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IFPair& p) const
    {
        return m_fValue > p.m_fValue;
    }
};

/*
Structure of pair
- point index
- counter
*/

struct IIPair
{
    int m_iIndex;
    int	m_iValue;

	IIPair()
	{
	    m_iIndex	= 0;
	    m_iValue	= 0;
	}

	IIPair(int p_iIndex, int p_iCounter)
	{
	    m_iIndex	= p_iIndex;
	    m_iValue	= p_iCounter;
	}

	// Overwrite operation < to get top K largest entries
    bool operator<(const IIPair& p) const
    {
        return m_iValue < p.m_iValue;
    }

    bool operator>(const IIPair& p) const
    {
        return m_iValue > p.m_iValue;
    }
};



// Define global options
extern int PARAM_DATA_N; // Number of points (rows) of X
extern int PARAM_QUERY_Q; // Number of rows (queries) of Q
extern int PARAM_DATA_D; // Number of dimensions

// Internal params
extern bool PARAM_INTERNAL_SAVE_OUTPUT; // save the results

// MIPS
extern int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
extern int PARAM_MIPS_TOP_B; // number of points to compute dot products
extern int PARAM_MIPS_NUM_SAMPLES; // Budgeted number of points to compute dot products


extern const char *X_FILE_PATH;
extern const char *Q_FILE_PATH;

#endif // HEADER_H_INCLUDED
