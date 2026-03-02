#pragma once
#include <cuda_runtime.h>
#include "../hyperblock/HyperBlock.h"
#include <stdio.h>
#include <limits>

#ifndef HyperBlockCuda_CUH
#define HyperBlockCuda_CUH
// === KERNELS: ===
__global__ void mergerHyperBlocks(
    const int seedIndex, int *readSeedQueue, const int numBlocks,
    const int numAttributes, const int numPoints, const float *opposingPoints,
    float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable);

__global__ void rearrangeSeedQueue(
    int *readSeedQueue, int *writeSeedQueue, int *deleteFlags,int *mergable, const int numBlocks);

__global__ void resetMergableFlags(int *mergableFlags, const int numBlocks);


__global__ void assignPointsToBlocks(
    const float *dataPointsArray, const int numAttributes, const int numPoints,
    const float *blockMins, const float *blockMaxes, const int *blockEdges,
    const int numBlocks, int *dataPointBlocks);


// === WRAPPERS: USE THESE TO CALL THE KERNELS FROM A CPP FILE!  ===
// ------------------------ creating hyperblocks wrapper functions --------------------------------
void mergerHyperBlocksWrapper(
    const int seedIndex, int *readSeedQueue, const int numBlocks,
    const int numAttributes, const int numPoints, const float *opposingPoints,
    float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable, int gridSize, int blockSize, int sharedMemSize);

void rearrangeSeedQueueWrapper(const int deadSeedNum, int *readSeedQueue, int *writeSeedQueue, int *deleteFlags,int *mergable, const int numBlocks, int gridSize, int blockSize);

void resetMergableFlagsWrapper(int *mergableFlags, const int numBlocks, int gridSize, int blockSize);



// --------------------------- removing useless blocks wrapper functions -------------------------
void assignPointsToBlocksWrapper(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks, int gridSize, int blockSize);

void sumPointsPerBlockWrapper(int *dataPointBlocks, const int numPoints, int *numPointsInBlocks, int gridSize, int blockSize);

void findBetterBlocksWrapper(const float *dataPointsArray, const int numAttributes, const int numPoints, const float *blockMins, const float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks, int *numPointsInBlocks, int gridSize, int blockSize);

__global__ void removeUselessAttributes(float* mins, float* maxes, const int* intervalCounts, const int minMaxLen, const int* blockEdges, const int numBlocks, const int* blockClasses, char* attrRemoveFlags, const int fieldLen, const float* dataset, const int numPoints, const int* classBorder, const int numClasses, const int *attributeOrder);
// same as above version, but this one assumes that the HBs are only one rule per attribute. so we can use a lot more efficient methods.
// makes changes to block bounds directly in the kernel, no need for the dumb flags.
__global__ void removeUselessAttributesNoDisjunctions(float *mins, float *maxes, const int numBlocks, const int FIELD_LENGTH, const int *blockClasses, const float *dataset, const int numPoints, const int *classBorder, const int *attributeOrder);
#endif