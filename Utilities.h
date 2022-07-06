#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "header.h"
#include <vector>
#include <queue>
#include <random>
#include <fstream> // fscanf, fopen, ofstream
#include <sstream>
#include <algorithm> // set_intersect(), lower_bound()
#include <unordered_map>
#include <unordered_set>

/**
Convert an integer to string
**/
inline string int2str(int x)
{
    stringstream ss;
    ss << x;
    return ss.str();
}

void extract_max_TopK(const Ref<VectorXf>&, int, Ref<VectorXi>);

// Output
void outputFile(const Ref<const MatrixXi> & , string );

// Printing
void printVector(const vector<int> & );
void printVector(const vector<IFPair> &);

#endif // UTILITIES_H_INCLUDED
