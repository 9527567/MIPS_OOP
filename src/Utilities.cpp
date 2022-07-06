#include "Utilities.h"



/** \brief Return top K index from a vector
 *
 * \param
 *
 - VectorXf::vecQuery:
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - VectorXi::vecTopK of top-k Index of largest value
 *
 */
void extract_max_TopK(const Ref<VectorXf> &p_vecQuery, int p_iTopK, Ref<VectorXi> p_vecTopK)
{
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (int n = 0; n < p_vecQuery.size(); ++n)
    {
        float fValue = p_vecQuery(n);

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(n, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(n, fValue));
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

void printVector(const vector<int> & vecPrint)
{
	cout << "Vector is: ";
	for (const auto &element: vecPrint)
		cout << element << " ";

	cout << endl;
}

void printVector(const vector<IFPair> & vecPrint)
{
	cout << "Vector is: ";
	for (const auto &element: vecPrint)
		cout << "{ " << element.m_iIndex << " " << element.m_fValue << " }, ";

	cout << endl;
}




/**
Input:
(col-wise) matrix p_matKNN of size K x Q

Output: Q x K
- Each row is for each query
**/
void outputFile(const Ref<const MatrixXi> & p_matKNN, string p_sOutputFile)
{
	cout << "Outputing File..." << endl;
	ofstream myfile(p_sOutputFile);

	//cout << p_matKNN << endl;

	for (int j = 0; j < p_matKNN.cols(); ++j)
	{
        //cout << "Print col: " << i << endl;
		for (int i = 0; i < p_matKNN.rows(); ++i)
		{
            myfile << p_matKNN(i, j) << ' ';

		}
		myfile << '\n';
	}

	myfile.close();
	cout << "Done" << endl;
}
