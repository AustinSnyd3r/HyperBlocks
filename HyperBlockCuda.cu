#include "HyperBlockCuda.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
// ------------------------------------------------------------------------------------------------
// REFACTORED MERGER HYPER BLOCKS KERNEL FUNCTION. DOESN'T NEED THE COOPERATIVE GROUPS.
// WRAP IN A LOOP. launch mergerHyperBlocks with i up to N - 1 as seed index, each time then rearrange, then reset.
// ------------------------------------------------------------------------------------------------
#define min(a, b) (a > b)? b : a
#define max(a, b) (a > b)? a : b
__global__ void mergerHyperBlocks(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float* __restrict__ points, float *hyperBlockMins, float *hyperBlockMaxes, int* deleteFlags, int* mergable, float* combinedMins, float* combinedMaxes){
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  	// Put the seed block attributes in instead.
	extern __shared__ float seedBlockAttributes[];

    float *seedBlockMins = &seedBlockAttributes[0];
    float *seedBlockMaxes = &seedBlockAttributes[numAttributes];
	const int totalThreadCnt = blockDim.x * gridDim.x;

    float *thisBlockCombinedMins = &combinedMins[threadID * numAttributes];
    float *thisBlockCombinedMaxes = &combinedMaxes[threadID * numAttributes];

    // get our seed block.
    const int seedBlock = readSeedQueue[seedIndex];

    // put seed block into shared mem
    const int baseIndex = seedBlock * numAttributes;
    for (int index = threadIdx.x; index < numAttributes; index += blockDim.x){
        int globalIndex = index + baseIndex;
    	//seedBlockMins[index] = hyperBlockMins[seedBlock * numAttributes + index];
        seedBlockMins[index] = hyperBlockMins[globalIndex];
        seedBlockMaxes[index] = hyperBlockMaxes[globalIndex];
    }

	// sync block so shared mem is right.
    __syncthreads();

    int k = threadID;
    while(k < numBlocks){

            // make the combined mins and maxes, and then check against all our data.
            if (k < numBlocks && k != seedBlock && deleteFlags[k] != -1 ){

                // first we build our combined list.
                for (int i = 0; i < numAttributes; i++){
                    thisBlockCombinedMaxes[i] = max(seedBlockMaxes[i], hyperBlockMaxes[k * numAttributes + i]);
                    thisBlockCombinedMins[i] = min(seedBlockMins[i], hyperBlockMins[k  * numAttributes + i]);
                }

                // now we check all our data for a point falling into our new bounds.
                char allPassed = 1;
                for (int point = 0; point < numPoints; point++){

                    char someAttributeOutside = 0;
                    for(int att = 0; att < numAttributes; att++){
                        const float val = points[point * numAttributes + att];
                        if (val > thisBlockCombinedMaxes[att] || val < thisBlockCombinedMins[att]){
                            someAttributeOutside = 1;
                            break;
                        }
                    }
                    // if there's NOT some attribute outside, this point has fallen in, and we can't do the merge.
                    if (!someAttributeOutside){
                        allPassed = 0;
                        break;
                    }
                }
                // if we did pass all the points, that means we can merge, and we can set the updated mins and maxes for this point to be the combined attributes instead.
                // then we simply flag that seedBlock is trash.
                if (allPassed){
                    // copy the combined mins and maxes into the original array
                    int index = k * numAttributes;
                    for (int i = 0; i < numAttributes; i++, index++){
                        hyperBlockMins[index] = thisBlockCombinedMins[i];
                        hyperBlockMaxes[index] = thisBlockCombinedMaxes[i];
                    }
                    // set the flag to -1.
                    deleteFlags[seedBlock] = -1;
                    mergable[k] = 1;
                }
            }

            // Move the threads to their new HyperBlock
            k += totalThreadCnt;
        }
}

__global__ void rearrangeSeedQueue(const int deadSeedNum, int *readSeedQueue, int *writeSeedQueue, int *deleteFlags, int *mergable, const int numBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;
    // now we are just going to loop through the seed queue and compute each blocks new position in the queue.
    for (int i = threadID; i < numBlocks; i += globalThreadCount) {
        
        if (i <= deadSeedNum){
            writeSeedQueue[i] = readSeedQueue[i];
            continue;
        }

        const int seed = readSeedQueue[i];

        if (mergable[seed] == 0) {
            int newIndex = deadSeedNum + 1;
            // Count only live, non-merged blocks in positions after deadSeedNum.
            for (int j = deadSeedNum + 1; j < i; j++) {
                int other = readSeedQueue[j];
                if (mergable[other] == 0)
                    newIndex++;
            }
            writeSeedQueue[newIndex] = seed;
        }
        // if we DID MERGE.
        else {
            int count = 0;
            // Count live merged blocks among indices after deadSeedNum.
            for (int j = deadSeedNum + 1; j < i; j++) {
                int other = readSeedQueue[j];
                if (mergable[other] == 1)
                    count++;
            }
            writeSeedQueue[numBlocks - 1 - count] = seed;
        }
    }
}

__global__ void resetMergableFlags(int *mergableFlags, const int numBlocks){
    // make all the flags 0.
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < numBlocks; i += gridDim.x * blockDim.x){
        mergableFlags[i] = 0;
    }
}

// ------------------------------------------------------------------------------------------------
// REMOVING USELESS HYPERBLOCKS KERNEL FUNCTION.
// RETAINS THE LOGIC FOR A DISJUNCTIVE BLOCK.
// LAUNCH 4 KERNELS!!! ASSIGN -> SUM -> FIND BETTER -> SUM. Once we have the count back, just delete all the HB's with 0 points remaining.
// ------------------------------------------------------------------------------------------------

// ASSIGN POINTS TO BLOCKS KERNEL FUNCTION.
// once every point has been assigned to a block, then we can start doing our removing of useless blocks.
__global__ void assignPointsToBlocks(const float *dataPointsArray, const int numAttributes, const int numPoints, float *blockMins, float *blockMaxes, const int *blockEdges, const int numBlocks, int *dataPointBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    const float *thisThreadPoint = &dataPointsArray[threadID * numAttributes];

    int *dataPointBlock;       // pointer to the slot in the numPoints long array that tells us which block this point goes into.
    float *startOfBlockMins;    // pointer to the start of the mins of the current block.
    float *startOfBlockMaxes;  // pointer to the start of the maxes of the current block.
    float *endOfBlock;         // pointer to the end of the current block.
    int currentBlock;          // the current block we are on.
    int nextBlock;             // the next block. useful because blocks are varying sizes.

    for(int i = threadID; i < numPoints; i += globalThreadCount){

        currentBlock = 0;
        nextBlock = 1;

        // set out pointer to where we assign this point to a block.
        dataPointBlock = &dataPointBlocks[i];
        *dataPointBlock = -1;

        thisThreadPoint = &dataPointsArray[i * numAttributes];

        // now we iterate through all the blocks. checking which block this point falls into first.
        while (currentBlock < numBlocks){

            // set up our start of mins and maxes.
            startOfBlockMins = &blockMins[blockEdges[currentBlock]];
            startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
            endOfBlock = &blockMins[blockEdges[nextBlock]];
            // now, we iterate through all the blocks, and the first one our point falls into, we set that block as the value of dataPointBlock, if not we put -1 and we have a coverage issue

            bool inThisBlock = true;

            // the x we are at, x0, x1, ...
            int particularAttribute = 0;

            // check through all the attributes for this block.
            while(startOfBlockMins < endOfBlock){

                // get the amount of x1's that we have in this particular block
                int countOfThisAttribute = (int)*startOfBlockMins;

                // increment these two at the same time, since they have the same length and same encoding of number of attributes in them
                startOfBlockMins++;
                startOfBlockMaxes++;

                // now loop that many times, checking if the point is in bounds of any of those intervals
                // we don't actually use i here, because we don't want to check the next attribute of our point on accident. since we may have 2 x2's and such.
                bool inBounds = false;
                for(int i = 0; i < countOfThisAttribute; i++){

                    const double min = *startOfBlockMins;
                    startOfBlockMins++;

                    const double max = *startOfBlockMaxes;
                    startOfBlockMaxes++;

                    const double pointValue = thisThreadPoint[particularAttribute];

                    // this loop is for the disjunctive blocks. if there is just one x, it doesn't matter. when we have 4 x2's to consider, once we are in one of them, we are done.
                    if(pointValue >= min && pointValue <= max){
                        inBounds = true;
                        break;
                    }
                }
                if (!inBounds){
                    inThisBlock = false;
                    break;
                }
                particularAttribute++;
            }
            // if in this block, we can set dataPointBlock and we're done
            if (inThisBlock){
                *dataPointBlock = currentBlock;
                break;
            }
            // increment the currentBlock and the next block.
            currentBlock++;
            nextBlock++;
        }
    }
}

// NOW OUR FUNCTION WHICH SUMS UP THE AMOUNT OF POINTS PER BLOCK
/*

Ryans algo. not fixing since we not using yet.


__global__ void sumPointsPerBlock(const int *dataPointBlocks, const int numPoints, int *numPointsInBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    for(int i = threadID; i < numPoints; i += globalThreadCount){
        atomicInc(&numPointsInBlocks[dataPointBlocks[i]]);
    }
}
__global__ void findBetterBlocks(int *dataPointsBlocks, const int numPoints, const int numBlocks, const int numAttributes, const float *points, const int *blockEdges, float *hyperBlockMins, float *hyperBlockMaxes, int *numPointsInBlocks){

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalThreadCount = gridDim.x * blockDim.x;

    float *startOfBlockMins;
    float *startOfBlockMaxes;
    float *endOfBlock;
    int *dataPointBlock;
    for(int i = threadID; i < numPoints; i += globalThreadCount){

        int currentBlock = dataPointsBlocks[i];
        int nextBlock = currentBlock + 1;
        dataPointBlock = &dataPointsBlocks[i];

        // we have a coverage issue if this happens.
        if (currentBlock == -1){
            continue;
        }

        float *thisThreadPoint = &points[i * numAttributes];

        // largest block size is our current block we are assigned to for this point
        int largestBlockSize = numPointsInBlocks[currentBlock];

        // now we iterate through and finally assign our point to the most populous block we find that we fit into.
        while (currentBlock < numBlocks){

            // now, we iterate through all the blocks after the one we chose, and if we find a bigger one we fit into, we go into that one.
            startOfBlockMins = &hyperBlockMins[blockEdges[currentBlock]];
            startOfBlockMaxes = &hyperBlockMaxes[blockEdges[currentBlock]];
            endOfBlock = &blockMins[blockEdges[nextBlock]];

            bool inThisBlock = true;

            // the x we are at, x0, x1, ...
            int particularAttribute = 0;

            // check through all the attributes for this block.
            while(startOfBlockMins < endOfBlock){

                // get the amount of x1's that we have in this particular block
                int countOfThisAttribute = (int)*startOfBlockMins;

                // increment these two at the same time, since they have the same length and same encoding of number of attributes in them
                startOfBlockMins++;
                startOfBlockMaxes++;

                // now loop that many times, checking if the point is in bounds of any of those intervals
                bool inBounds = false;
                for(int att = 0; att < countOfThisAttribute; att++){

                    const double min = *startOfBlockMins;
                    startOfBlockMins++;

                    const double max = *startOfBlockMaxes;
                    startOfBlockMaxes++;

                    const double pointValue = thisThreadPoint[particularAttribute];

                    if (pointValue >= min && pointValue <= max && numPointsInBlocks[currentBlock] > largestBlockSize){
                        inBounds = true;
                        break;
                    }
                }
                if (!inBounds){
                    inThisBlock = false;
                    break;
                }
                particularAttribute++;
            }
            if (inThisBlock){
                *dataPointBlock = currentBlock;
                largestBlockSize = numPointsInBlocks[currentBlock];
            }
            currentBlock++;
            nextBlock++;
        }
    }
}
*/

void mergerHyperBlocksWrapper(const int seedIndex, int *readSeedQueue, const int numBlocks, const int numAttributes, const int numPoints, const float *opposingPoints,float *hyperBlockMins, float *hyperBlockMaxes, int *deleteFlags, int *mergable, int gridSize, int blockSize, int sharedMemSize, float* combinedMins, float* combinedMaxes){
	mergerHyperBlocks<<<gridSize, blockSize, sharedMemSize>>>(
            seedIndex,
            readSeedQueue,
            numBlocks,
            numAttributes,
            numPoints,
            opposingPoints,
	    	hyperBlockMins,
			hyperBlockMaxes,
			deleteFlags,
			mergable,
      		combinedMins,
      		combinedMaxes
		);
    return;
}

void rearrangeSeedQueueWrapper(const int deadSeedCount, int *readSeedQueue, int *writeSeedQueue, int *deleteFlags, int *mergable, const int numBlocks, int gridSize, int blockSize){
    rearrangeSeedQueue<<<gridSize, blockSize>>>(deadSeedCount, readSeedQueue, writeSeedQueue, deleteFlags, mergable, numBlocks);
	return;
}
void resetMergableFlagsWrapper(int *mergableFlags, const int numBlocks, int gridSize, int blockSize){
	resetMergableFlags<<<gridSize, blockSize>>>(mergableFlags, numBlocks);
  	return;
}

/**
* IMPLEMENT ME NOT YET DONE!!!!!!!!!!
*/
void assignPointsToBlocksWrapper(const float *dataPointsArray, const int numAttributes, const int numPoints,const float *blockMins, const float *blockMaxes, const int *blockEdges,const int numBlocks, int *dataPointBlocks, int gridSize, int blockSize){
	return;
}