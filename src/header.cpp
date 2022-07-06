#include "header.h"

/**
Defining the default values of global variables
*/

// Input
int PARAM_DATA_N = 0 ; // Number of points (rows) of X
int PARAM_QUERY_Q = 0; // Number of rows (queries) of Q
int PARAM_DATA_D = 0; // Number of dimensions

// Internal params
bool PARAM_INTERNAL_SAVE_OUTPUT = true; // save the results

// MIPS
int PARAM_MIPS_TOP_K = 0; // TopK largest entries from Xq
int PARAM_MIPS_TOP_B = 0; // number of points to compute dot products
int PARAM_MIPS_NUM_SAMPLES = 10000; // Budgeted number of points to compute dot products

//data file path
const char *X_FILE_PATH;
const char *Q_FILE_PATH;
