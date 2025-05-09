//
// Created by asnyd on 3/20/2025.
//
#include "Simplifications.h"
int Simplifications::REMOVAL_COUNT = 0;

// runs our three kernel functions which remove useless blocks.
void Simplifications::removeUselessBlocks(std::vector<std::vector<std::vector<float>>> &data, std::vector<HyperBlock>& hyper_blocks) {
    /*
     * The algorithm to remove useless blocks does basically this.
     *     - take one particular point in our dataset. Find the first HB that it fits into.
     *     - then, once everyone has found their first choice, we sum up the count of which HBs have how many points
     *     - then, we run it again, this time starting from the block which each point chose. We pick a new HB instead, if we find one which our point falls into, and which has a higher amount of points in it than our current
     *     - this is not a perfect way of doing it, but at least allows us to find the "most general blocks" based on the count of how many points are in each. This way we can then just delete whichever blocks we find with no *UNIQUE* points in them.
     *     * notice how we are putting all data in, and all blocks together. this allows us to find errors as well. we may find that a block is letting in wrong class points this way.
     */
    int FIELD_LENGTH = data[0][0].size();
    int datasetSize = 0;
    for (auto &c : data)
        datasetSize += c.size();

    std::vector<std::vector<float>> minMaxResult = DataUtil::flattenMinsMaxesForRUB(hyper_blocks, FIELD_LENGTH);
    std::vector<std::vector<float>> flattenedData =  DataUtil::flattenDataset(data);

    // Use references to avoid copying.
    const std::vector<float>& blockMins   = minMaxResult[0];
    const std::vector<float>& blockMaxes  = minMaxResult[1];

    // Cast each element from the third std::vector (floats) into ints.
    const std::vector<float> &edgesAsFloats = minMaxResult[2];
    std::vector<int> blockEdges;
    blockEdges.resize(minMaxResult[2].size());
    // cast result [2] to ints, since this is the block edges. the array which tells us where each block starts and ends (as indexes).
    std::transform(edgesAsFloats.begin(), edgesAsFloats.end(), blockEdges.begin(),
              [](float val) -> int { return static_cast<int>(val); });

    // Get the dataPointsArray (again using a reference).
    const std::vector<float>& dataPointsArray = flattenedData[0];

    const int numPoints = dataPointsArray.size() / FIELD_LENGTH;
    std::vector<int> dataPointBlocks(numPoints, 0);              // Each point's chosen block.
    const int numBlocks = hyper_blocks.size();                    // Number of hyperblocks.
    std::vector<int> numPointsInBlocks(numBlocks, 0);              // Count of points in each hyperblock.

    // Allocate device memory and copy data.
    float *d_dataPointsArray, *d_blockMins, *d_blockMaxes;
    int   *d_blockEdges;
    int *d_dataPointBlocks, *d_numPointsInBlocks;

    cudaMalloc((void**)&d_dataPointsArray, sizeof(float) * numPoints * FIELD_LENGTH);
    cudaMemcpy(d_dataPointsArray, dataPointsArray.data(), sizeof(float) * numPoints * FIELD_LENGTH, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_blockMins, sizeof(float) * blockMins.size());
    cudaMemcpy(d_blockMins, blockMins.data(), sizeof(float) * blockMins.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_blockMaxes, sizeof(float) * blockMaxes.size());
    cudaMemcpy(d_blockMaxes, blockMaxes.data(), sizeof(float) * blockMaxes.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_blockEdges, sizeof(int) * blockEdges.size());
    cudaMemcpy(d_blockEdges, blockEdges.data(), sizeof(int) * blockEdges.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_dataPointBlocks, sizeof(int) * numPoints);
    cudaMemset(d_dataPointBlocks, 0, sizeof(int) * numPoints);

    cudaMalloc((void**)&d_numPointsInBlocks, sizeof(int) * numBlocks);
    cudaMemset(d_numPointsInBlocks, 0, sizeof(int) * numBlocks);

    // Determine grid and block sizes using CUDA occupancy.
    int minGridSize, blockSize;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, assignPointsToBlocks, 0, 0);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaOccupancyMaxPotentialBlockSize: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    assignPointsToBlocksWrapper(d_dataPointsArray, FIELD_LENGTH, numPoints, d_blockMins, d_blockMaxes, d_blockEdges, numBlocks, d_dataPointBlocks, gridSize, blockSize);
    cudaDeviceSynchronize();

    sumPointsPerBlockWrapper(d_dataPointBlocks, numPoints, d_numPointsInBlocks, gridSize, blockSize);
    cudaDeviceSynchronize();

    findBetterBlocksWrapper(d_dataPointsArray, FIELD_LENGTH, numPoints, d_blockMins, d_blockMaxes, d_blockEdges, numBlocks, d_dataPointBlocks, d_numPointsInBlocks, gridSize, blockSize);
    cudaDeviceSynchronize();

    // Reset the numPointsInBlocks array on the device, this is because we have now found better homes, and we are ready to recompute the sums.
    cudaMemset(d_numPointsInBlocks, 0, sizeof(int) * numBlocks);
    sumPointsPerBlockWrapper(d_dataPointBlocks, numPoints, d_numPointsInBlocks, gridSize, blockSize);
    cudaDeviceSynchronize();

    // Copy back the computed numPointsInBlocks.
    cudaMemcpy(numPointsInBlocks.data(), d_numPointsInBlocks, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree((void *)d_dataPointsArray);
    cudaFree((void *)d_blockMins);
    cudaFree((void *)d_blockMaxes);
    cudaFree((void *)d_blockEdges);
    cudaFree((void *)d_dataPointBlocks);
    cudaFree((void *)d_numPointsInBlocks);

    // Remove blocks with less than our count of unique points
    // unique points refers to the amount of points which are classified uniquely by this particular block.
    for (int i = numPointsInBlocks.size() - 1; i >= 0; i--) {
        if (numPointsInBlocks[i] <= REMOVAL_COUNT)
            hyper_blocks.erase(hyper_blocks.begin() + i);
    }
}

void Simplifications::removeUselessAttr(std::vector<HyperBlock>& hyper_blocks, std::vector<std::vector<std::vector<float>>>& data, std::vector<std::vector<int>>& attributeOrderings) {
    int FIELD_LENGTH = data[0][0].size();

    // Prepare host data by flattening your data structures.
    auto fMinMaxResult =  DataUtil::flatMinMaxNoEncode(hyper_blocks, FIELD_LENGTH);
    auto fDataResult =  DataUtil::flattenDataset(data);

    // Build host arrays from the flattened results:
    std::vector<float> mins = fMinMaxResult[0];
    std::vector<float> maxes = fMinMaxResult[1];
    int minMaxLen = static_cast<int>(mins.size());

    std::vector<int> blockEdges(fMinMaxResult[2].size());
    for (size_t i = 0; i < fMinMaxResult[2].size(); i++) {
        blockEdges[i] = static_cast<int>(fMinMaxResult[2][i]);
    }
    int numBlocks = static_cast<int>(hyper_blocks.size());

    std::vector<int> blockClasses(fMinMaxResult[3].size());
    for (size_t i = 0; i < fMinMaxResult[3].size(); i++) {
        blockClasses[i] = static_cast<int>(fMinMaxResult[3][i]);
    }

    std::vector<int> intervalCounts(fMinMaxResult[4].size());
    for (size_t i = 0; i < fMinMaxResult[4].size(); i++) {
        intervalCounts[i] = static_cast<int>(fMinMaxResult[4][i]);
    }

    // Create flags array (initialize to 0).
    std::vector<char> attrRemoveFlags(hyper_blocks.size() * FIELD_LENGTH, 0);

    // Prepare the dataset.
    std::vector<float> dataset = fDataResult[0];
    int numPoints = static_cast<int>(dataset.size() / FIELD_LENGTH);

    std::vector<int> classBorder(fDataResult[1].size());
    for (size_t i = 0; i < fDataResult[1].size(); i++) {
        classBorder[i] = static_cast<int>(fDataResult[1][i]);
    }
    int numClasses = static_cast<int>(hyper_blocks.size());

    std::vector<int> attributeOrderingsFlattened(attributeOrderings.size() * FIELD_LENGTH, 0);
    for (int i = 0; i < attributeOrderings.size(); i++) {
        copy(attributeOrderings[i].begin(), attributeOrderings[i].end(),
            attributeOrderingsFlattened.begin() + i * FIELD_LENGTH);
    }

    // Device pointers.
    float* d_mins = nullptr;
    float* d_maxes = nullptr;
    int* d_intervalCounts = nullptr;
    int* d_blockEdges = nullptr;
    int* d_blockClasses = nullptr;
    char* d_attrRemoveFlags = nullptr;
    float* d_dataset = nullptr;
    int* d_classBorder = nullptr;
    int *d_attributeOrderingsFlattened = nullptr;

    // Allocate device memory.
    cudaMalloc((void**)&d_mins, mins.size() * sizeof(float));
    cudaMalloc((void**)&d_maxes, maxes.size() * sizeof(float));
    cudaMalloc((void**)&d_intervalCounts, intervalCounts.size() * sizeof(int));
    cudaMalloc((void**)&d_blockEdges, blockEdges.size() * sizeof(int));
    cudaMalloc((void**)&d_blockClasses, blockClasses.size() * sizeof(int));
    cudaMalloc((void**)&d_attrRemoveFlags, attrRemoveFlags.size() * sizeof(char));
    cudaMalloc((void**)&d_dataset, dataset.size() * sizeof(float));
    cudaMalloc((void**)&d_classBorder, classBorder.size() * sizeof(int));
    cudaMalloc((void**)&d_attributeOrderingsFlattened, attributeOrderingsFlattened.size() * sizeof(int));

    // Copy host data to device.
    cudaMemcpy(d_mins, mins.data(), mins.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxes, maxes.data(), maxes.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intervalCounts, intervalCounts.data(), intervalCounts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockEdges, blockEdges.data(), blockEdges.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockClasses, blockClasses.data(), blockClasses.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attrRemoveFlags, attrRemoveFlags.data(), attrRemoveFlags.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataset, dataset.data(), dataset.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_classBorder, classBorder.data(), classBorder.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attributeOrderingsFlattened, attributeOrderingsFlattened.data(), attributeOrderingsFlattened.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Determine execution configuration.
    int blockSize;
    int gridSize;

    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, mergerHyperBlocks, 0, 0);
    gridSize = (numBlocks + blockSize - 1) / blockSize;

    // Launch the kernel.
    removeUselessAttributesWrapper(d_mins, d_maxes, d_intervalCounts, minMaxLen, d_blockEdges, numBlocks, d_blockClasses, d_attrRemoveFlags, FIELD_LENGTH, d_dataset, numPoints, d_classBorder, numClasses, d_attributeOrderingsFlattened, gridSize, blockSize);
    cudaDeviceSynchronize();

    // Copy results from device (flags) back to host.
    cudaMemcpy(attrRemoveFlags.data(), d_attrRemoveFlags, attrRemoveFlags.size() * sizeof(char), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_mins);
    cudaFree(d_maxes);
    cudaFree(d_intervalCounts);
    cudaFree(d_blockEdges);
    cudaFree(d_blockClasses);
    cudaFree(d_attrRemoveFlags);
    cudaFree(d_dataset);
    cudaFree(d_classBorder);
    cudaFree(d_attributeOrderingsFlattened);

    // Update the hyper_blocks based on the flags.
    for (size_t hb = 0; hb < hyper_blocks.size(); hb++) {
        HyperBlock &block = hyper_blocks[hb];
        // For each attribute in the block (assumes FIELD_LENGTH attributes per block)
        for (int attr = 0; attr < FIELD_LENGTH; attr++) {
            int index = hb * FIELD_LENGTH + attr;
            if (attrRemoveFlags[index] == 1) {
                // Remove the attribute intervals and reset to default values.
                if (attr < block.minimums.size() && attr < block.maximums.size()) {
                    block.minimums[attr].clear();
                    block.maximums[attr].clear();
                    block.minimums[attr].push_back(0.0f);
                    block.maximums[attr].push_back(1.0f);
                }
            }
        }
    }
}

std::vector<int> Simplifications::runSimplifications(std::vector<HyperBlock> &hyperBlocks, std::vector<std::vector<std::vector<float>>> &trainData, std::vector<std::vector<int>> &bestAttributeOrderings){
    int FIELD_LENGTH = trainData[0][0].size();
    int runCount = 0;
    int totalClauses = 0;
    int updatedClauses = 0;

    //do{
        // set our count of what we have to start
        totalClauses = updatedClauses;
        runCount++; // counter so we can show how many iterations this took.

        // simplification functions
        Simplifications::removeUselessAttr(hyperBlocks, trainData, bestAttributeOrderings);
        //Simplifications::removeUselessBlocks(trainData, hyperBlocks);

        // count how many we have after simplifications.
        updatedClauses = 0;
        for(HyperBlock &hyperBlock : hyperBlocks) {
            for(int i = 0; i < FIELD_LENGTH; i++){
                if (hyperBlock.minimums[i][0] == 0 && hyperBlock.maximums[i][0] == 1.0f){
                    continue;
                }
                else
                    updatedClauses += hyperBlock.minimums[i].size();
            }

        }

    //// iteratively call the simplifications until we don't remove any more clauses.
//} while(updatedClauses != totalClauses);
    return { runCount, updatedClauses };
}
