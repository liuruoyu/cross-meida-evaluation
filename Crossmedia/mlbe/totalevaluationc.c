/*
 //
 //  map_evaluationc.c
 //  
 //
 //  Created by Zhen Yi on 18/6/12.
 //  Copyright 2012 HKUST. All rights reserved.
 //
 */

#include <math.h>
#include "mex.h"

/* Input Arguments */

#define	Dtrue_IN	prhs[0] /* matrix of true neighbors */
#define	Dhamm_IN	prhs[1] /* matrix of hamming distance */
#define	r_IN	prhs[2] /* hamming radius */
#define	k_IN	prhs[3] /* hamming rank position */

/* Output Arguments */

#define	rpre_OUT	plhs[0] /* for hamming radius evaluation: precision */
#define	rrec_OUT	plhs[1] /* for hamming radius evaluation: recall */
#define	rret_OUT	plhs[2] /* for hamming radius evaluation: retrieved examples */
#define	kpre_OUT	plhs[3] /* for hamming rank evaluation: precision */
#define	krec_OUT	plhs[4] /* for hamming rank evaluation: recall */
#define	kmap_OUT	plhs[5] /* for hamming rank evaluation: map */

/*
 #if !defined(MAX)
 #define	MAX(A, B)	((A) > (B) ? (A) : (B))
 #endif
 
 #if !defined(MIN)
 #define	MIN(A, B)	((A) < (B) ? (A) : (B))
 #endif
 
 static	double	mu = 1/82.45;
 static	double	mus = 1 - 1/82.45;
 */

static void bubblesort (unsigned short *topKlabels, unsigned short *hammingvector, unsigned short *labelvector, int K, int totallength) {
    /* find top K labels with the smallest hamming distance */
    int i, j;
    unsigned short *tphammingvector = (unsigned short*) malloc(sizeof(unsigned short)*totallength);
    /* mexPrintf("\n number of elements: %ld.\n", totallength); */
    for (i=0;i<totallength;i++) {
        *(tphammingvector+i) = *(hammingvector+i);
    }
    for (i=0; i<K; i++) {
        unsigned short iradius = 1000;
        int tpposision, iposition = -1;
        for (j=0; j<totallength;j++){
            if (*(tphammingvector+j)< iradius) {
                iposition = j;
                iradius = *(tphammingvector+iposition);
            }
        }        
        topKlabels[i] = *(labelvector+iposition);
        *(tphammingvector+iposition) = 1000;
    }
    free(tphammingvector);
    return;
}

static double get_precision ( unsigned short	*labelvector, int labellength) {
    double iprecision = 0.0;
    int id=0;
    for (id = 0; id < labellength; id++) {
        if (*(labelvector+id)>0) {
            iprecision++;
        }
    }
    mxAssert(labellength>0, "divide zero when computing precision");
    iprecision /= (double)labellength;
    return iprecision;
}

static double get_recall (unsigned short *returnedlabels, int rankposition, int allgoods) {
    double irecall = 0.0;
    int id=0;
    for (id = 0; id < rankposition; id++) {
        if (*(returnedlabels+id)>0) {
            irecall++;
        }
    }
    if (allgoods > 0) {
        irecall /= (double)allgoods;
    }
    else {
        irecall = 1;
    }
    
    return irecall;
}

static double get_ap( unsigned short	*labelvector, int labellength)
{
    /* labelvector is a vector of length rankposition */
    double ap = 0.0;
    double NumOfGoodpoint = 0.0;
    double CurrentPrecision = 0.0;
    int id=0;
    for (id = 0; id< labellength; id++) {
        if (*(labelvector+id)>0) {
            CurrentPrecision = (NumOfGoodpoint+1)/(id+1);
            ap = (ap*NumOfGoodpoint+CurrentPrecision)/(NumOfGoodpoint+1);
            NumOfGoodpoint++;
        }
    }
    return ap;
}

static void totalevaluationc(
                            double *rpre, double *rrec, double *rret,
                            double *kpre, double *krec, double *kmap,
                            mxLogical *W, unsigned short *D,
                            double radiusposition, double rankposition,
                            int mrows, int ncols
                            )
{
    long rad = 0, n=0, allgoods=0, count = 0, trueneighbor=0, queryid=0;
    long long numel = (long long) mrows*ncols;

    double total_good_pairs = 0;
    double *retrieved_all_pairs = (double *) malloc(sizeof(double)*(radiusposition+1));
    double *retrieved_good_pairs = (double *) malloc(sizeof(double)*(radiusposition+1));
    
    double totalprecision=0.0, totalrecall=0.0, totalap=0.0;
    unsigned short *topKlabels = (unsigned short *) malloc(sizeof(unsigned short) * rankposition);
    unsigned short *ihammingvector = (unsigned short *) malloc(sizeof(unsigned short) * mrows);
    unsigned short *ilabelvector = (unsigned short *) malloc(sizeof(unsigned short) * mrows);
    double *precisions = (double *) malloc(sizeof(double)*(ncols));
    double *recalls = (double *) malloc(sizeof(double)*(ncols));
    double *aps = (double *) malloc(sizeof(double)*(ncols));
    
    
    /* initialize parameters mexPrintf("\n no. of elements: %ld.\n", numel); */
    
    for (rad = 0; rad <= radiusposition; rad++) {
        retrieved_good_pairs[rad] = 0.0;
        retrieved_all_pairs[rad] = 0.0;
    }

    for (rad = 0; rad < ncols; rad++) {
        precisions[rad] = 0.0;
        recalls[rad] = 0.0;
        aps[rad] = 0.0;
    }    
    for (rad=0; rad < mrows; rad++) {
        ihammingvector[rad] = 0;
        ilabelvector[rad] = 0;
    }    
    for (rad = 0; rad < rankposition; rad++) {
        topKlabels[rad] = 0;
    }/**/
    
    /* Approach 1 */
    while (numel--) {
        /*mexPrintf("\n no. of elements: %ld.\n", numel);*/
        unsigned short currentHamm = *D++;
        mxLogical currentLabel = *W++;
        /*mexPrintf("\n current point: %ld %ld.\n", currentHamm, currentLabel);*/
        
        for (rad = 0; rad <= radiusposition; rad++) {
            if (currentHamm <= rad) {
                retrieved_all_pairs[rad]++;
                if (currentLabel > 0) {
                    retrieved_good_pairs[rad]++ ;
                }
            }
        }
        if (currentLabel > 0) {
            total_good_pairs++;
        }
        
        if (count < mrows) {
            ihammingvector[count] = currentHamm;
            ilabelvector[count] = (unsigned short)currentLabel;
            if (currentLabel>0) {
                trueneighbor++;
            }
            count++;
        } else {
            /* process last query*/
            if (trueneighbor >0) {
                for (rad = 0; rad < rankposition; rad++) {
                    topKlabels[rad] = 0;
                }
                bubblesort(topKlabels, ihammingvector, ilabelvector, rankposition, mrows); 
                /*for (rad=0; rad<mrows;rad++) {
                    mexPrintf("%ld ", ihammingvector[rad]);
                }mexPrintf("\n");
                for (rad=0; rad<rankposition;rad++) {
                    mexPrintf("%ld ", topKlabels[rad]);
                }mexPrintf("\n");*/
                precisions[queryid] = get_precision(topKlabels, rankposition);
                recalls[queryid] = get_recall(topKlabels, rankposition, trueneighbor);
                aps[queryid] = get_ap(topKlabels, rankposition);
                 /*
                   mexPrintf("\n precision(%ld):%f.\n", queryid, precisions[queryid]);
                   mexPrintf("\n recall(%ld):%f.\n", queryid, recalls[queryid]);
                   mexPrintf("\n ap(%ld):%f.\n", queryid, aps[queryid]);*/
            } else {
                precisions[queryid] = 1.0;
                recalls[queryid] = 1.0;
                aps[queryid] = 1.0; 
            }
            totalprecision += precisions[queryid];
            totalrecall += recalls[queryid];
            totalap += aps[queryid];
            
            /* clear temp variables for next query */
            for (rad=0; rad < mrows; rad++) {
                ihammingvector[rad] = 0;
                ilabelvector[rad] = 0;
            }
            count = 0;
            trueneighbor=0;
            queryid++;
            
            /* process current point */
            ihammingvector[count] = currentHamm;
            ilabelvector[count] = (unsigned short)currentLabel;
            if (currentLabel>0) {
                trueneighbor++;
            }
            count++;
        }
    }
    
    /* for the last query */
    if (count>0) {
        if (trueneighbor >0) {
            for (rad = 0; rad < rankposition; rad++) {
                topKlabels[rad] = 0;
            }
            bubblesort(topKlabels, ihammingvector, ilabelvector, rankposition, mrows); 
            /*for (rad=0; rad<mrows;rad++) {
                mexPrintf("%ld ", ihammingvector[rad]);
            }mexPrintf("\n");
            for (rad=0; rad<rankposition;rad++) {
                mexPrintf("%ld ", topKlabels[rad]);
            }mexPrintf("\n");*/
            precisions[queryid] = get_precision(topKlabels, rankposition);
            recalls[queryid] = get_recall(topKlabels, rankposition, trueneighbor);
            aps[queryid] = get_ap(topKlabels, rankposition);
        } else {
            precisions[queryid] = 1.0;
            recalls[queryid] = 1.0;
            aps[queryid] = 1.0; 
        }
        /*mexPrintf("\n ap(%ld):%f.\n", queryid, aps[queryid]);*/
        totalprecision += precisions[queryid];
        totalrecall += recalls[queryid];
        totalap += aps[queryid];
        queryid++;
    }
    
    mexPrintf("\n Processed Queries: %ld \n", queryid);
    
    /*
     for (rad=0; rad<mrows*ncols; rad++) {
     bool ilabel = *(W+rad);
     mexPrintf("\n Dtrue(%ld,%ld):%ld.\n", mrows, ncols, ilabel);
     }
     unsigned short ihammdis = *(D+mrows*(1000));
     bool ilabel = *(W+mrows*(10));
     mexPrintf("\n Dhamm(%ld,%ld):%ld.\n", mrows, ncols, ihammdis);
     mexPrintf("\n Dtrue(%ld,%ld):%ld.\n", mrows, ncols, ilabel);*/ 
    
    
    /* Approach 2
     for (n = 0; n<ncols; n++) {
     allgoods = 0;
     for (rad = 0; rad < rankposition; rad++) {
     topKlabels[rad] = 0;
     }
     
     for (rad = 0; rad < mrows; rad++) {
     if (*(W + n*mrows + rad)>0) {
     allgoods++;
     }
     }
     if (allgoods >0) {
     bubblesort(topKlabels, (D + n*mrows), (W + n*mrows), rankposition, mrows);
     
     precisions[n] = get_precision(topKlabels, rankposition);
     recalls[n] = get_recall(topKlabels, rankposition, allgoods);
     aps[n] = get_ap(topKlabels, rankposition); 
     
     } else {
     precisions[n] = 1.0;
     recalls[n] = 1.0;
     aps[n] = 1.0; 
     }
     
     totalprecision += precisions[n];
     totalrecall += recalls[n];
     totalap += aps[n];
     }  */
    
    /* compute values for ouput*/
    *kpre = totalprecision/(double)ncols;
    *krec = totalrecall/(double)ncols;
    *kmap = totalap/(double)ncols;
    /*
    mexPrintf("\n mean precision:%f.\n", *kpre);
    mexPrintf("\n mean recall:%f.\n", *krec);
    mexPrintf("\n mean average precision:%f.\n", *kmap);*/
    
    for (rad = 0; rad <= radiusposition; rad++) {
        if (retrieved_all_pairs[rad] > 0.01)
            *(rpre+rad) = retrieved_good_pairs[rad]/retrieved_all_pairs[rad];
        else
            *(rpre+rad) = 0.0;
        *(rrec+rad) = retrieved_good_pairs[rad]/total_good_pairs;
        *(rret+rad) = retrieved_all_pairs[rad];
    }
    
    /* free memory*/
    free(retrieved_all_pairs);
    free(retrieved_good_pairs);
    free(topKlabels);
    free(ihammingvector);
    free(ilabelvector);
    free(precisions);
    free(recalls);
    free(aps);
    
    return;
}


void mexFunction( int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray*prhs[] )

{ 
    mxLogical *Wmat;
    unsigned short *Dmat;
    double r, k; 
    double *rprevalue, *rrecvalue, *rretvalue, *kprevalue, *krecvalue, *kmapvalue;
    int mrows, ncols;
    
    /* Check for proper number of arguments */
    
    if (nrhs != 4) { 
        mexErrMsgTxt("Three input arguments required."); 
    } else if (nlhs != 6) {
        mexErrMsgTxt("Three output arguments required."); 
    } 
    
    /* Each column is for a test query 
     
     numel = mxGetNumberOfElements(Dtrue_IN);*/ 
    mrows = mxGetM(Dtrue_IN); 
    ncols = mxGetN(Dtrue_IN);
    
    /*int64_T tempn = (long long)5000*995000;mexPrintf("\nThe number of elements is %ld.\n", tempn);
    mexPrintf("\n rows: %ld\tcols: %ld.\n", mxGetM(Dtrue_IN), mxGetN(Dtrue_IN));
    mexPrintf("\n rows: %ld\tcols: %ld.\n", mxGetM(Dhamm_IN), mxGetN(Dhamm_IN));
    mexPrintf("\nThe number of elements are %ld and %ld.\n", mxGetNumberOfElements(Dtrue_IN), mxGetNumberOfElements(Dhamm_IN));*/
    
     /*
     // numel2 = mxGetNumberOfElements(Dhamm_IN);
     // mexPrintf("\nThe number of elements are %ld and %ld.\n", numel, numel2);
     //
     //     
     if (!mxIsDouble(Y_IN) || mxIsComplex(Y_IN) || 
     (MAX(m,n) != 4) || (MIN(m,n) != 1)) { 
     mexErrMsgTxt("YPRIME requires that Y be a 4 x 1 vector."); 
     } */
    
    /* Assign pointers to the input parameters */     
    Wmat = mxGetLogicals(Dtrue_IN); 
    Dmat = (unsigned short *)mxGetData(Dhamm_IN); 
    r = mxGetScalar(r_IN);
    k = mxGetScalar(k_IN);
    
    /* Assign pointers to the output parameters */     
    rpre_OUT = mxCreateDoubleMatrix(1, r+1, mxREAL);
    rrec_OUT = mxCreateDoubleMatrix(1, r+1, mxREAL);
    rret_OUT = mxCreateDoubleMatrix(1, r+1, mxREAL);
    kpre_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
    krec_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
    kmap_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    rprevalue = mxGetPr(rpre_OUT); 
    rrecvalue = mxGetPr(rrec_OUT);
    rretvalue = mxGetPr(rret_OUT);
    kprevalue = mxGetPr(kpre_OUT); 
    krecvalue = mxGetPr(krec_OUT);
    kmapvalue = mxGetPr(kmap_OUT);
    
    /* Do the actual computations in a subroutine */
    totalevaluationc(rprevalue, rrecvalue, rretvalue, kprevalue, krecvalue, kmapvalue, Wmat, Dmat, r, k, mrows, ncols); 
    return;
    
}



