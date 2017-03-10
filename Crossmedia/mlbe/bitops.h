#ifndef MATCH_H__
#define MATCH_H__

#include <stdio.h>
#include <math.h>
#include "types.h"

const int lookup [] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

inline int match(UINT8*P, UINT8*Q, int codelb) {
	switch(codelb) {
		case 4:
			return __builtin_popcount(*(UINT32*)P ^ *(UINT32*)Q);
		break;
		case 8:
			return __builtin_popcountll(*(UINT64*)P ^ *(UINT64*)Q);
		break;
		case 16:
			return    __builtin_popcountll(((UINT64*)P)[0] ^ ((UINT64*)Q)[0]) \
				+ __builtin_popcountll(((UINT64*)P)[1] ^ ((UINT64*)Q)[1]);
		break;
		case 32:
			return    __builtin_popcountll(((UINT64*)P)[0] ^ ((UINT64*)Q)[0]) \
				+ __builtin_popcountll(((UINT64*)P)[1] ^ ((UINT64*)Q)[1]) \
				+ __builtin_popcountll(((UINT64*)P)[2] ^ ((UINT64*)Q)[2]) \
				+ __builtin_popcountll(((UINT64*)P)[3] ^ ((UINT64*)Q)[3]);
		break;
		default:
			int output = 0;
			for (int i=0; i<codelb; i++) 
				output+= lookup[P[i] ^ Q[i]];
			return output;
		break;
	}
}

void split (UINT8* code, int codelb, UINT64*chunks, int nchunks, int chunkl) {
	UINT64 temp = 0x0;
	int nbits = 0, nbyte = 0;
	UINT64 mask = chunkl==64? 0xFFFFFFFFFFFFFFFF : ((UINT64_1 << chunkl) - UINT64_1);
	for (int i=0; i<nchunks; i++) {
		while (nbits < chunkl && nbyte < codelb) {
			temp |= ((UINT64)code[nbyte]<<nbits);
			nbits += 8; nbyte ++;
		}
		chunks[i] = temp & mask;
		temp = chunkl==64? temp==0x0: temp >> chunkl; 
		nbits -= chunkl;
	}
}


#endif
