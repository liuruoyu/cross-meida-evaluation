#include <mex.h>
#include <stdio.h>
#include <math.h>
#include "bitops.h"

/*
dist = hammD2 (mat1, mat2)

Input: 
   mat1, mat2: compact uint8 bit vectors. Each binary code is one column.
   size(mat1) = [nwords, ndatapoints1]
   size(mat2) = [nwords, ndatapoints2]

Output:
   dist:	Hamming distances between columns of mat1 and mat2
   size(dist) = [ndatapoints1, ndatapoints2]

/* Input Arguments */
#define	MAT1   	prhs[0]  // Uint8 vector of size n x 1
#define MAT2    prhs[1]  // Uint8 matrix of size n x m

/* Output Arguments */
#define	OUTPUT	plhs[0]  // Double vector 1 x m binary hamming distance


void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    int nWords, nVecs1, nVecs2, nWords_Vecs,  i, j;
    unsigned int dist;
    UINT16* outputp;
    unsigned char tmp, *pMat2, *pMat1, *pV;
    
    /* Check for proper number of arguments */    
    if (nrhs != 2) { 
	mexErrMsgTxt("Two input arguments required."); 
    }
    
    if (!mxIsUint8(MAT2))
	mexErrMsgTxt("Matrix must be uInt8");
    
    if (!mxIsUint8(MAT1))
	mexErrMsgTxt("Vector must be uInt8");
    
    /* Get dimensions of inputs */
    nWords = (int) mxGetM(MAT2); 
    nVecs1 = (int) mxGetN(MAT2); 
    nWords_Vecs = (int)  mxGetM(MAT1); 
    nVecs2 = (int) mxGetN(MAT1);
    
    if (nWords!=nWords_Vecs)
	mexErrMsgTxt("Dimension mismatch btw. matrix and vector");
    
    // Create output array
    OUTPUT = mxCreateNumericMatrix(nVecs2, nVecs1, mxUINT16_CLASS, mxREAL);
    outputp = (UINT16*) mxGetPr(OUTPUT);
    
    pMat1 = (unsigned char*) mxGetPr(MAT1);
    pMat2 = (unsigned char*) mxGetPr(MAT2);
    
    for (i=0; i<nVecs1; i++)
	for (j=0; j<nVecs2; j++)
	    *(outputp++) = match(&pMat1[j*nWords], &pMat2[i*nWords], nWords);
    
    return;    
}
