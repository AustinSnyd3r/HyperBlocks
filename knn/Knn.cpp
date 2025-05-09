//
// Created by asnyd on 3/20/2025.
//

#include "Knn.h"
#include <vector>
#include <utility>
#include <queue>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <algorithm>


constexpr float EPSILON = 1e-6f;

// Lets make a K-nn that goes through the unclassified points and sees how close they are to being
// inside of each of the blocks. If the value for a attribute is within the bounds of the block we wont add any
// distance to the sum. If the value is outside the bounds we will add the distance to the sum.
std::vector<std::vector<long>> Knn::closeToInkNN(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int k, int NUM_CLASSES) {
    // Basically we will do the same thing, we will just need to change our distancce thingy around.
    int FIELD_LENGTH = hyperBlocks[0].maximums.size();
    if (k > hyperBlocks.size()) k = (int)sqrt(hyperBlocks.size());

    // Keep track of assignments with something
    std::vector<std::vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
      classifications[i] = std::vector<float>(unclassifiedData[i].size());    // Put the std::vector for each class
    }

    #pragma omp parallel for
    for(int i = 0; i < NUM_CLASSES; i++){
        #pragma omp parallel for
        // For each point in unclassified points
        for(int point = 0; point < unclassifiedData[i].size(); point++){
            // Use a priority queue to keep track of the top k best distances
            std::priority_queue<std::pair<float, int>> kNearest;

            // Go through all the blocks and find the disstances to their centers
            for(const HyperBlock& hyperBlock : hyperBlocks){
                // Find the distance between the HB center and the unclassified data point
                
                float distance =  hyperBlock.distance_to_HB_Avg(FIELD_LENGTH, unclassifiedData[i][point].data());

                if(kNearest.size() < k){    // always add when queue is not at k yet.
                    kNearest.push(std::make_pair(distance, hyperBlock.classNum));
                }
                else if(distance < kNearest.top().first){ // Queue is big enough, and this distance is better than the worst in queue
                    kNearest.pop();    // pop the max (worst distance)
                    kNearest.push(std::make_pair(distance, hyperBlock.classNum));    // push the better distance.
                }
            }

           std::vector<float> weightedVotes(NUM_CLASSES, 0.0);
		   while(!kNearest.empty()){
				float dist = kNearest.top().first;
    			int cls = kNearest.top().second;
   				kNearest.pop();

   				 float weight = (dist == 0) ? 1.0 : (1.0 / pow(dist, 2));  // Inverse squared weight
    			weightedVotes[cls] += weight;
			}

			// Assign the class with the highest weighted vote
			int majorityClass = std::distance(weightedVotes.begin(), std::max_element(weightedVotes.begin(), weightedVotes.end()));

            classifications[i][point] = majorityClass;
        }
    }

    std::vector<std::vector<long>> regularConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    // Go through the classes.
    for(int classN = 0; classN < NUM_CLASSES; classN++){
        for(int point = 0; point < classifications[classN].size(); point++){
            regularConfusionMatrix[classN][classifications[classN][point]]++;
        }
    }

    return regularConfusionMatrix;
}

////
////
std::vector<std::vector<long>> Knn::closestBlock(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<HyperBlock>& hyperBlocks, int NUM_CLASSES){


    int FIELD_LENGTH = hyperBlocks[0].maximums.size();

    // Keep track of assignments with something
    std::vector<std::vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
        classifications[i] = std::vector<float>(unclassifiedData[i].size());    // Put the std::vector for each class
    }

    for(int i = 0; i < NUM_CLASSES; i++){

        #pragma omp parallel for
        // For each point in unclassified points
        for(int point = 0; point < unclassifiedData[i].size(); point++){

            // we take the closest HB. But, there is some caveats. We must consider some things. Imagine missing an interval by .10, small miss. But if that interval itself is
            // only .05 wide, meaning it is a very tight interval, then missing by that distance is actually huge. In this case, we have missed by 200% the width of the interval. that is what we add.
            // now we take the block with the smallest distance as a ratio.
            float bestDistanceAsRatio = std::numeric_limits<float>::infinity();
            // another consideration. Let's imagine that out of 10 intervals in this simplified HB, 8 are removed. And we had a total miss of 200%. That means that we have missed badly on the only
            // intervals which even matter. So in this case, we don't want to take that guy. So we need to also consider the amount of intervals which 'count'.
            int bestClass = -1;

            // Go through all the blocks and find the distances to their centers
            for(const HyperBlock& hyperBlock : hyperBlocks){
                float currentDistanceRatio = 0.0f;
                int numIntervalsNotRemoved = 0;

                for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {

                    // att is value of our attribute in that point
                    float att = unclassifiedData[i][point][attribute];

                    // best distance ratio in most cases, is just the ratio for that attribute. but we have to allow for disjunctions.
                    float bestAttributeDistanceRatio = std::numeric_limits<float>::infinity();

                    // mini loop. if we didn't allow disjunctive units, this wouldn't need to be a loop.
                    for (int c = 0; c < hyperBlock.minimums[attribute].size(); c++) {

                        // NEEDS FIXING FOR DISJUNCTIONS
                        if (hyperBlock.minimums[attribute][c] != 0.0f || hyperBlock.maximums[attribute][c] != 1.0f) {
                            numIntervalsNotRemoved++;
                        }

                        // take distance to closer edge.
                        float distance;
                        if (att < hyperBlock.minimums[attribute][c]) {
                            distance = hyperBlock.minimums[attribute][c] - att;
                        } else if (att > hyperBlock.maximums[attribute][c]) {
                            distance = att - hyperBlock.maximums[attribute][c];
                        } else {
                            bestAttributeDistanceRatio = 0.0f;
                            break;
                        }

                        float intervalWidth = hyperBlock.maximums[attribute][c] - hyperBlock.minimums[attribute][c];
                        bestAttributeDistanceRatio = std::min(distance / intervalWidth, bestAttributeDistanceRatio);
                    }
                    // if there was a positive distance. we increment the count of missed attributes, and then increase ratio.
                    currentDistanceRatio += bestAttributeDistanceRatio;
                }

                // divide the distance ratio by the number of interals we were out of. So that we get an "average miss ratio"
                currentDistanceRatio /= numIntervalsNotRemoved;
                if (currentDistanceRatio < bestDistanceAsRatio) {
                    bestDistanceAsRatio = currentDistanceRatio;
                    bestClass = hyperBlock.classNum;
                }
            }
            classifications[i][point] = bestClass;
        }
    }

    std::vector<std::vector<long>> regularConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    // Go through the classes.
    for(int classN = 0; classN < NUM_CLASSES; classN++){
        for(int point = 0; point < classifications[classN].size(); point++){
            regularConfusionMatrix[classN][classifications[classN][point]]++;
        }
    }

    return regularConfusionMatrix;
}




std::vector<std::vector<long>> Knn::pureKnn(std::vector<std::vector<std::vector<float>>> unclassifiedData, std::vector<std::vector<std::vector<float>>> classifiedData, int NUM_CLASSES, int k) {

    int FIELD_LENGTH = classifiedData[0][0].size();

    // This will hold the predicted class for each unclassified point
    std::vector<std::vector<int>> classifications(NUM_CLASSES); // [trueClass][pointIndex]
    for (int i = 0; i < NUM_CLASSES; i++) {
        classifications[i] = std::vector<int>(unclassifiedData[i].size(), -1);
    }

    // For each unclassified point
    for (int trueClass = 0; trueClass < NUM_CLASSES; ++trueClass) {
        for (int u = 0; u < unclassifiedData[trueClass].size(); ++u) {

            std::priority_queue<std::pair<float, int>> kNearest; // {distance, classLabel}

            // Compare to all classified points
            for (int cClass = 0; cClass < NUM_CLASSES; ++cClass) {
                for (int c = 0; c < classifiedData[cClass].size(); ++c) {
                    float distance = Knn::euclideanDistancePoints(
                        unclassifiedData[trueClass][u], classifiedData[cClass][c], FIELD_LENGTH);

                    if (kNearest.size() < k) {
                        kNearest.push({distance, cClass});
                    } else if (distance < kNearest.top().first) {
                        kNearest.pop();
                        kNearest.push({distance, cClass});
                    }
                }
            }

            // Tally up class votes
            std::vector<int> votes(NUM_CLASSES, 0);
            while (!kNearest.empty()) {
                int predictedClass = kNearest.top().second;
                votes[predictedClass]++;
                kNearest.pop();
            }

            // Determine majority vote
            int majorityClass = -1;
            int maxVotes = -1;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                if (votes[c] > maxVotes) {
                    maxVotes = votes[c];
                    majorityClass = c;
                }
            }

            classifications[trueClass][u] = majorityClass;
        }
    }

    // Build confusion matrix
    std::vector<std::vector<long>> confusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));
    for (int trueClass = 0; trueClass < NUM_CLASSES; ++trueClass) {
        for (int p = 0; p < classifications[trueClass].size(); ++p) {
            int predictedClass = classifications[trueClass][p];
            confusionMatrix[trueClass][predictedClass]++;
        }
    }

    return confusionMatrix;
}


/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
std::vector<std::vector<long>> Knn::kNN(
    std::vector<std::vector<std::vector<float>>> unclassifiedData,
    std::vector<HyperBlock>& hyperBlocks,
    int k,
    int NUM_CLASSES
) {
    int FIELD_LENGTH = hyperBlocks[0].maximums.size();
    std::cout << "Field Length: " << FIELD_LENGTH << std::endl;

    if (k > hyperBlocks.size()) k = (int)std::sqrt(hyperBlocks.size());

    std::vector<std::vector<float>> classifications(NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES; i++) {
        classifications[i] = std::vector<float>(unclassifiedData[i].size());
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int point = 0; point < unclassifiedData[i].size(); point++) {
            // Store all distances
            std::vector<std::pair<float, int>> allDistances;

            for (const auto& hyperBlock : hyperBlocks) {
                float bottomDist = Knn::euclideanDistanceBounds(hyperBlock.tamedMin, unclassifiedData[i][point], FIELD_LENGTH);
                float topDist = Knn::euclideanDistanceBounds(hyperBlock.tamedMax, unclassifiedData[i][point], FIELD_LENGTH);

                float distance = std::min(bottomDist, topDist);
                allDistances.emplace_back(distance, hyperBlock.classNum);
            }

            // Get the k closest elements
            std::nth_element(allDistances.begin(), allDistances.begin() + k, allDistances.end());
            std::vector<std::pair<float, int>> kNearest(allDistances.begin(), allDistances.begin() + k);

            // Find max distance in kNearest to ignore
            auto worstIt = std::max_element(kNearest.begin(), kNearest.end());
            kNearest.erase(worstIt);  // remove worst

            // Count votes excluding the farthest
            std::vector<int> votes(NUM_CLASSES, 0);
            for (const auto& pair : kNearest) {
                votes[pair.second]++;
            }

            int majorityClass = -1;
            int maxVotes = 0;

            for (int c = 0; c < NUM_CLASSES; c++) {
                if (votes[c] > maxVotes) {
                    maxVotes = votes[c];
                    majorityClass = c;
                }
            }

            classifications[i][point] = majorityClass;
        }
    }

    std::vector<std::vector<long>> regularConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));
    for (int classN = 0; classN < NUM_CLASSES; classN++) {
        for (int point = 0; point < classifications[classN].size(); point++) {
            regularConfusionMatrix[classN][(int)classifications[classN][point]]++;
        }
    }

    return regularConfusionMatrix;
}


//EUCLIDEAN DISTANCE OF TWO VECTORS, comparing a point to a block bound (2-D vector for disjunctions)
float Knn::euclideanDistanceBounds(const std::vector<float>& blockBound, const std::vector<float>& point, int FIELD_LENGTH){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = blockBound[i] - point[i];
        sumSquaredDifference += diff * diff;
    }

    return sqrt(sumSquaredDifference);
}


//EUCLIDEAN DISTANCE OF TWO VECTORS, comparing a point to a point
float Knn::euclideanDistancePoints(const std::vector<float>& point2, const std::vector<float>& point, int FIELD_LENGTH){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = point2[i] - point[i];
        sumSquaredDifference += diff * diff;
    }

    return sqrt(sumSquaredDifference);
}

/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
std::vector<std::vector<long>> Knn::mergableKNN(std::vector<std::vector<std::vector<float>>> &unclassifiedData, std::vector<std::vector<std::vector<float>>> &trainingData, std::vector<HyperBlock> &hyperBlocks, int NUM_CLASSES) {

    int FIELD_LENGTH = trainingData[0][0].size();

    // our confusion matrix
    // make each vector for each point
    std::vector<std::vector<int>> predictedLabels(NUM_CLASSES);
    for (int actualClass = 0; actualClass < NUM_CLASSES; actualClass++) {
        predictedLabels[actualClass].resize(unclassifiedData[actualClass].size());
    }

    // now we just take our training data, and first break it up by column exactly how we did in in the interval HB generation portion.
    std::vector<std::vector<DataATTR>> dataByAttribute = IntervalHyperBlock::separateByAttribute(trainingData, FIELD_LENGTH);

    for (HyperBlock &block : hyperBlocks) {
        // now we set up the top and bottom pairs for each block just to make sure that the bounds are made.
        block.topBottomPairs.resize(FIELD_LENGTH);

        // For each attribute (column), perform a binary search. we are just making the bounds of each attribute for each HB in index form.
        for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {

            // Get the sorted vector of DataATTR for this attribute.
            const std::vector<DataATTR>& columnPoints = dataByAttribute[attribute];

            // Our target interval for this attribute (assuming each is stored as a one-element vector)
            float lowerVal = block.minimums[attribute][0];
            float upperVal = block.maximums[attribute][0];

            // lower_bound: first element whose value is not less than lowerVal.
            auto lowIt = lower_bound(columnPoints.begin(), columnPoints.end(), lowerVal,
                [](const DataATTR &d, float value) {
                    return d.value < value;
            });

            // upper_bound: first element whose value is greater than upperVal.
            auto highIt = upper_bound(columnPoints.begin(), columnPoints.end(), upperVal,
                [](float value, const DataATTR &d) {
                    return value < d.value;
            });

            // Determine indices:
            // If lowIt reached the end, then no elements are ≥ lowerVal.
            int lowIndex = (lowIt != columnPoints.end()) ? static_cast<int>(lowIt - columnPoints.begin()) : -1;

            // For the topmost index, we want the last index in the range.
            // upper_bound returns the first element greater than upperVal.
            // If highIt equals columnPoints.begin(), then even the first element is greater than upperVal.
            int highIndex = (highIt != columnPoints.begin()) ? static_cast<int>(highIt - columnPoints.begin()) - 1 : -1;

            // Store the pair of indices (bottommost, topmost) for this attribute.
            block.topBottomPairs[attribute] = std::make_pair(lowIndex, highIndex);
        }
    }

    // now we can go through the list of unclassified points, and pick the block which is most legally mergeable to our new point
    for (int actualClass = 0; actualClass < NUM_CLASSES; actualClass++) {
        auto &pointsInClass = unclassifiedData[actualClass];

        // for each point of each class.
        for (int pIndex = 0; pIndex < pointsInClass.size(); pIndex++) {
            auto &point = pointsInClass[pIndex];

            // tells us which index this point would be in in the sorted columns if inserted. we don't actually put it in though, that would change the data.
            std::vector<int> pointIndicesByColumn(FIELD_LENGTH, -1);

            // set up our vector which tells us which indices the point would land at.
            for (int attribute = 0; attribute < FIELD_LENGTH; attribute++) {

                float pointVal = point[attribute];

                const std::vector<DataATTR> &columnPoints = dataByAttribute[attribute];
                // find the index which our point would go at in this column.
                auto indexIt = std::lower_bound(
                    columnPoints.begin(), columnPoints.end(), pointVal,
                    [](const DataATTR &d, float value) { return d.value < value; });

                int index = std::distance(columnPoints.begin(), indexIt);
                if (index == columnPoints.size())
                    index--;
                pointIndicesByColumn[attribute] = index;
            }

            int bestClass = -1;
            float bestAcc = 0.0f;
            int bestBlockSize = 0;
            #pragma omp parallel
            for (HyperBlock &block : hyperBlocks) {

                // now that our block is made, we can determine how many wrong points are going to fall in if we included this guy.
                // use the merge check to determine how "mergeable" this point is to each block, and we use the "most mergeable" class.
                float acc = mergeCheck(pointIndicesByColumn, block, dataByAttribute);

                // if the accuracy is better, or within .1% and we have a bigger block, we are using this block.
                if (acc > bestAcc || (std::fabs(acc - bestAcc) < .001f && block.size > bestBlockSize) ) {
                    bestClass = block.classNum;
                    bestAcc = acc;
                    bestBlockSize = block.size;
                }
            }

            predictedLabels[actualClass][pIndex] = bestClass;
        }
    }

    // build and return the confusion matrix
    std::vector<std::vector<long>> confusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    for (int actualClass = 0; actualClass < NUM_CLASSES; actualClass++) {
        for (int pIdx = 0; pIdx < predictedLabels[actualClass].size(); pIdx++) {
            int predictedClass = predictedLabels[actualClass][pIdx];
            confusionMatrix[actualClass][predictedClass]++;
        }
    }

    return confusionMatrix;
}

// returns amount of wrong class points which would fall into our HB bounds if we did include this point in the block.
// doesn't change the block itself, just changes it hypothetically. we will call this the justin herbert algorithm.
// same logic as the checking mergeable we implemented for the interval Hyper without cuda merging stuff, but returns the wrong count instead of just true or false.
//  ───────────────────────────────────────────────────────────────
//  Evaluate the accuracy of HyperBlock `hb` if we enlarged it just
//  enough to include the unclassified point whose per‑column index
//  is given in insertIdx
//  ───────────────────────────────────────────────────────────────
float Knn::mergeCheck(std::vector<int> &insertIdx, HyperBlock &hb, std::vector<std::vector<DataATTR>> &columns) {
    const int D = columns.size();

    // 1.  Make a local copy of bounds and enlarge with the new point.
    std::vector<std::pair<int,int>> bounds = hb.topBottomPairs;
    for (int d = 0; d < D; ++d) {
        bounds[d].first  = std::min(bounds[d].first,  insertIdx[d]);
        bounds[d].second = std::max(bounds[d].second, insertIdx[d]);
    }

    // 2.  Choose the attribute with the SMALLEST interval → fewest candidates.
    int pivot = 0;
    std::size_t span = std::numeric_limits<std::size_t>::max();
    for (int d = 0; d < D; ++d) {
        std::size_t cur = bounds[d].second - bounds[d].first + 1;
        if (cur < span) { span = cur; pivot = d; }
    }

    std::size_t wrong = 0;

    // 3.  Scan candidate rows in the pivot column only.
    const auto& col = columns[pivot];
    for (int idx = bounds[pivot].first; idx <= bounds[pivot].second; ++idx) {
        const DataATTR& attr = col[idx];

        // Skip rows that already belong to the block’s class.
        if (attr.classNum == hb.classNum) continue;

        // Check every other dimension quickly; bail on first failure.
        bool inside = true;
        int row = attr.classIndex;
        for (int d = 0; d < D; ++d) {

            if (d == pivot)
                continue;

            float v = columns[d][row].value;
            bool ok = false;
            const auto& mins = hb.minimums[d];
            const auto& maxs = hb.maximums[d];

            for (std::size_t c = 0; c < mins.size(); ++c) {
                if (v >= mins[c] && v <= maxs[c]) {
                    ok = true;
                    break;
                }
            }

            // new point might have extended the bound
            if (!ok && v >= columns[d][insertIdx[d]].value && v <= columns[d][insertIdx[d]].value) ok = true;

            if (!ok) {
                inside = false;
                break;
            }
        }

        if (inside) ++wrong;
    }

    // 4.  Accuracy if we accept the point (+1 correct) and wrong extras.
    return float(hb.size + 1) / float(hb.size + 1 + wrong);
}


std::vector<std::vector<long>> Knn::bruteMergable(
    std::vector<std::vector<std::vector<float>>> unclassifiedData,
    std::vector<std::vector<std::vector<float>>> classifiedData,
    std::vector<HyperBlock>& hyperBlocks,
    int k,
    int NUM_CLASSES
) {
    constexpr float EPSILON = 1e-6f;
    constexpr int NUM_SAMPLES = 300; // Sampling limit for each class

    int FIELD_LENGTH = hyperBlocks[0].maximums.size();
    std::vector<std::vector<float>> classifications(NUM_CLASSES);

    for (int i = 0; i < NUM_CLASSES; i++) {
        classifications[i] = std::vector<float>(unclassifiedData[i].size());
    }

    // Pre-process hyperblocks by class for faster access
    std::vector<std::vector<size_t>> blocksByClass(NUM_CLASSES);
    for (size_t i = 0; i < hyperBlocks.size(); i++) {
        blocksByClass[hyperBlocks[i].classNum].push_back(i);
    }

    // Stores min and max average impurity for each class of blocks
    std::vector<std::pair<float, float>> classImpurityRange(NUM_CLASSES, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()});

    // Store average impurity metrics for each class combination
    std::vector<std::vector<float>> classMetrics(NUM_CLASSES, std::vector<float>(NUM_CLASSES, 0.0f));

    // Create a single vector of all classified points with their class labels for faster iteration
    struct ClassifiedPoint {
        std::vector<float> point;
        int classLabel;
    };
    std::vector<ClassifiedPoint> allClassifiedPoints;
    allClassifiedPoints.reserve(std::accumulate(
        classifiedData.begin(),
        classifiedData.end(),
        0,
        [](size_t sum, const auto& vec) { return sum + vec.size(); }
    ));

    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        for (const auto& point : classifiedData[cls]) {
            allClassifiedPoints.push_back({point, cls});
        }
    }

    // First phase: Gather metrics using sampling
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_CLASSES; i++) {
        // Pick samples for this class
        std::vector<std::vector<float>> classSamples;
        int sampleCount = std::min(NUM_SAMPLES, static_cast<int>(classifiedData[i].size()));

        // Create a copy of indices that we can shuffle
        std::vector<int> indices(classifiedData[i].size());
        std::iota(indices.begin(), indices.end(), 0);

        // Use a thread-safe random generator
        unsigned int seed = std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

        // Select the first sampleCount indices
        for (int j = 0; j < sampleCount; j++) {
            classSamples.push_back(classifiedData[i][indices[j]]);
        }

        // Thread-local metrics to avoid contention
        std::vector<std::pair<float, int>> threadMetrics(NUM_CLASSES, {0.0f, 0});

        // For each target class, calculate metrics
        for (int targetClass = 0; targetClass < NUM_CLASSES; targetClass++) {
            if (blocksByClass[targetClass].empty()) continue;

            float totalImpurity = 0.0f;
            int validMerges = 0;

            // Process each sample with each block of target class
            for (const auto& sample : classSamples) {
                for (size_t blockIdx : blocksByClass[targetClass]) {
                    const auto& block = hyperBlocks[blockIdx];

                    // Calculate expansion area efficiently
                    bool anyChanges = false;
                    std::vector<std::pair<float, float>> expansionRanges(FIELD_LENGTH, {0.0f, 0.0f});
                    std::vector<bool> hasExpansion(FIELD_LENGTH, false);

                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        if (sample[attr] < block.minimums[attr][0]) {
                            expansionRanges[attr].first = sample[attr];
                            expansionRanges[attr].second = block.minimums[attr][0];
                            hasExpansion[attr] = true;
                            anyChanges = true;
                        } else if (sample[attr] > block.maximums[attr][0]) {
                            expansionRanges[attr].first = block.maximums[attr][0];
                            expansionRanges[attr].second = sample[attr];
                            hasExpansion[attr] = true;
                            anyChanges = true;
                        }
                    }

                    if (!anyChanges) continue;

                    // Count points in expansion area
                    int rightCls = 0;
                    int wrongCls = 0;

                    for (const auto& cp : allClassifiedPoints) {
                        // Skip if point is already in the block
                        bool insideBlock = true;
                        for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                            if (cp.point[attr] < block.minimums[attr][0] - EPSILON || cp.point[attr] > block.maximums[attr][0] + EPSILON) {
                                insideBlock = false;
                                break;
                            }
                        }
                        if (insideBlock) continue;

                        // Check if point is in any expansion area
                        bool inExpansion = false;
                        for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                            if (hasExpansion[attr] &&
                                cp.point[attr] >= expansionRanges[attr].first - EPSILON &&
                                cp.point[attr] <= expansionRanges[attr].second + EPSILON) {
                                inExpansion = true;
                                break;
                            }
                        }

                        if (inExpansion) {
                            if (cp.classLabel == targetClass) {
                                rightCls++;
                            } else {
                                wrongCls++;
                            }
                        }
                    }

                    // Calculate impurity
                    if (rightCls + wrongCls > 0) {
                        float impurity = static_cast<float>(wrongCls) / (rightCls + wrongCls);
                        totalImpurity += impurity;
                        validMerges++;
                    }
                }
            }

            if (validMerges > 0) {
                threadMetrics[targetClass].first = totalImpurity;
                threadMetrics[targetClass].second = validMerges;
            }
        }

        // Merge thread-local results to global metrics
        #pragma omp critical
        {
            for (int targetClass = 0; targetClass < NUM_CLASSES; targetClass++) {
                if (threadMetrics[targetClass].second > 0) {
                    float avgImpurity = threadMetrics[targetClass].first / threadMetrics[targetClass].second;
                    classMetrics[targetClass][i] = avgImpurity;

                    // Update min/max range
                    if (avgImpurity < classImpurityRange[targetClass].first) {
                        classImpurityRange[targetClass].first = avgImpurity;
                    }
                    if (avgImpurity > classImpurityRange[targetClass].second) {
                        classImpurityRange[targetClass].second = avgImpurity;
                    }
                }
            }
        }
    }

    // Second phase: Classify unclassified data
    std::vector<std::vector<long>> results(NUM_CLASSES);

    // Pre-allocate results to avoid resizing
    for (int i = 0; i < NUM_CLASSES; i++) {
        results[i].reserve(unclassifiedData[i].size());
    }

    // Process each class of unclassified data in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
        // Thread-local results vector to avoid contention
        std::vector<long> threadResults;
        threadResults.reserve(unclassifiedData[classIdx].size());

        for (size_t pointIdx = 0; pointIdx < unclassifiedData[classIdx].size(); pointIdx++) {
            const auto& point = unclassifiedData[classIdx][pointIdx];
            std::vector<float> bestImpurities(NUM_CLASSES, std::numeric_limits<float>::max());

            // Check each class of blocks
            for (int targetClass = 0; targetClass < NUM_CLASSES; targetClass++) {
                if (blocksByClass[targetClass].empty()) continue;

                // Find best block for this point and target class
                for (size_t blockIdx : blocksByClass[targetClass]) {
                    const auto& block = hyperBlocks[blockIdx];

                    // If point is already inside the block, perfect score
                    bool insideBlock = true;
                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        if (point[attr] < block.minimums[attr][0] - EPSILON || point[attr] > block.maximums[attr][0] + EPSILON) {
                            insideBlock = false;
                            break;
                        }
                    }

                    if (insideBlock) {
                        bestImpurities[targetClass] = 0.0f;
                        break;
                    }

                    // Calculate expansion area efficiently
                    std::vector<std::pair<float, float>> expansionRanges(FIELD_LENGTH, {0.0f, 0.0f});
                    std::vector<bool> hasExpansion(FIELD_LENGTH, false);
                    bool anyChanges = false;

                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        if (point[attr] < block.minimums[attr][0]) {
                            expansionRanges[attr].first = point[attr];
                            expansionRanges[attr].second = block.minimums[attr][0];
                            hasExpansion[attr] = true;
                            anyChanges = true;
                        } else if (point[attr] > block.maximums[attr][0]) {
                            expansionRanges[attr].first = block.maximums[attr][0];
                            expansionRanges[attr].second = point[attr];
                            hasExpansion[attr] = true;
                            anyChanges = true;
                        }
                    }

                    if (!anyChanges) continue;

                    // Count points in expansion area
                    int rightCls = 0;
                    int wrongCls = 0;

                    for (const auto& cp : allClassifiedPoints) {
                        // Skip if point is already in the block
                        bool insideBlock = true;
                        for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                            if (cp.point[attr] < block.minimums[attr][0] - EPSILON || cp.point[attr] > block.maximums[attr][0] + EPSILON) {
                                insideBlock = false;
                                break;
                            }
                        }
                        if (insideBlock) continue;

                        // Check if point is in any expansion area
                        bool inExpansion = false;
                        for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                            if (hasExpansion[attr] &&
                                cp.point[attr] >= expansionRanges[attr].first - EPSILON &&
                                cp.point[attr] <= expansionRanges[attr].second + EPSILON) {
                                inExpansion = true;
                                break;
                            }
                        }

                        if (inExpansion) {
                            if (cp.classLabel == targetClass) {
                                rightCls++;
                            } else {
                                wrongCls++;
                            }
                        }
                    }

                    // Calculate impurity
                    float impurity = 1.0f; // Default to worst impurity
                    if (rightCls + wrongCls > 0) {
                        impurity = static_cast<float>(wrongCls) / (rightCls + wrongCls);
                    } else if (anyChanges) {
                        impurity = 0.5f; // Neutral score for empty expansion
                    }

                    bestImpurities[targetClass] = std::min(bestImpurities[targetClass], impurity);
                }
            }

            // Normalize scores and find best class
            std::vector<float> normalizedScores(NUM_CLASSES);
            for (int c = 0; c < NUM_CLASSES; c++) {
                float minImpurity = classImpurityRange[c].first;
                float maxImpurity = classImpurityRange[c].second;

                if (maxImpurity - minImpurity < EPSILON) {
                    normalizedScores[c] = (bestImpurities[c] < minImpurity + EPSILON) ? 0.0f : 1.0f;
                } else {
                    normalizedScores[c] = (bestImpurities[c] - minImpurity) / (maxImpurity - minImpurity);
                }
            }

            // Find class with best score
            int bestClass = 0;
            float bestScore = normalizedScores[0];
            for (int c = 0; c < NUM_CLASSES; c++) {
                if (normalizedScores[c] < bestScore) {
                    bestScore = normalizedScores[c];
                    bestClass = c;
                }
            }

            threadResults.push_back(bestClass);
            classifications[classIdx][pointIdx] = static_cast<float>(bestClass);
        }

        // Merge thread results into global results
        #pragma omp critical
        {
            results[classIdx].insert(results[classIdx].end(), threadResults.begin(), threadResults.end());
        }
    }

    return results;
}



bool Knn::isInside(const std::vector<float>& point, const std::vector<std::vector<float>>& fMins, const std::vector<std::vector<float>>& fMaxes){
    for (int i = 0; i < point.size(); ++i) {
        bool insideDim = false;

        // Go through all possible disjunctives
        for (int r = 0; r < fMins[i].size(); ++r) {
            if ((point[i] + EPSILON >= fMins[i][r]) && (point[i] - EPSILON <= fMaxes[i][r])) {
                insideDim = true;
                break;
            }
        }

        if (!insideDim) return false;
    }

    return true;
}

// returns the distance between two points, by attribute. This is helpful when we are not able to compare the distances by aggregating in any way. Useful for heterogenous data.
std::vector<float> Knn::losslessDistance(std::vector<float> &seedPoint, std::vector<float> &trainPoint) {
    std::vector<float> distances(seedPoint.size(), 0);
    for (int attribute = 0; attribute < seedPoint.size(); attribute++)
        distances[attribute] = std::fabs(seedPoint[attribute] - trainPoint[attribute]);

    return distances;
}

std::vector<float> Knn::computeStdDeviations(const std::vector<std::vector<std::vector<float>>>& trainData) {
    // total number of points
    int numPoints = 0;
    for (const auto& classPoints : trainData)
        numPoints += static_cast<int>(classPoints.size());
    if (numPoints == 0) return {};

    int FIELD_LENGTH = static_cast<int>(trainData[0][0].size());

    // compute means
    std::vector<float> means(FIELD_LENGTH, 0.0f);
    for (const auto& classPoints : trainData) {
        for (const auto& point : classPoints) {
            for (int j = 0; j < FIELD_LENGTH; ++j) {
                means[j] += point[j];
            }
        }
    }
    for (int j = 0; j < FIELD_LENGTH; ++j) {
        means[j] /= numPoints;
    }

    // compute sum of squared deviations
    std::vector<float> sqDiffs(FIELD_LENGTH, 0.0f);
    for (const auto& classPoints : trainData) {
        for (const auto& point : classPoints) {
            for (int j = 0; j < FIELD_LENGTH; ++j) {
                float d = point[j] - means[j];
                sqDiffs[j] += d * d;
            }
        }
    }

    // finalize to standard deviations
    std::vector<float> stddevs(FIELD_LENGTH);
    for (int j = 0; j < FIELD_LENGTH; ++j) {
        stddevs[j] = std::sqrt(sqDiffs[j] / numPoints);

        // if you want sample std-dev instead, use:
        // stddevs[j] = std::sqrt(sqDiffs[j] / (numPoints - 1));
    }
    return stddevs;
}

std::vector<std::vector<long>> Knn::thresholdKNN(std::vector<std::vector<std::vector<float>>> &unclassifiedData, std::vector<std::vector<std::vector<float>>> &classifiedData, int NUM_CLASSES, int k, float threshold) {

    int FIELD_LENGTH = classifiedData[0][0].size();

    // This will hold the predicted class for each unclassified point
    std::vector<std::vector<int>> classifications(NUM_CLASSES); // [trueClass][pointIndex]
    for (int i = 0; i < NUM_CLASSES; i++) {
        classifications[i] = std::vector<int>(unclassifiedData[i].size(), -1);
    }

    // our vector of standard deviations
    std::vector<float> deviations = computeStdDeviations(classifiedData);

    int totalTrainingPoints = 0;
    for (auto& classPoints : classifiedData) {
        totalTrainingPoints += classPoints.size();
    }

    // For each unclassified point
    for (int trueClass = 0; trueClass < NUM_CLASSES; ++trueClass) {

        for (int point = 0; point < unclassifiedData[trueClass].size(); ++point) {
            auto &unclassifiedPoint = unclassifiedData[trueClass][point];

            // Instead of a priority_queue, do this:
            std::vector<std::pair<int,int>> sims;
            sims.reserve(totalTrainingPoints);

            for (int trainClass = 0; trainClass < NUM_CLASSES; ++trainClass) {
                for (auto &p : classifiedData[trainClass]) {
                    int sim = 0;
                    std::vector<float> differences = Knn::losslessDistance(unclassifiedPoint, p);
                    for (int att = 0; att < FIELD_LENGTH; ++att)
                        if (differences[att] < deviations[att] * threshold)
                            ++sim;
                    sims.emplace_back(sim, trainClass);
                }
            }

            // 2) sort descending
            std::sort(sims.begin(), sims.end(),
                      [](std::pair<int,int> &a, std::pair<int,int> &b){ return a.first > b.first; });

            // 3) find cutoff similarity
            int cutoffIdx = std::min((int)sims.size()-1, k-1);
            int cutoffSim = sims[cutoffIdx].first;

            // 4) vote on *all* sims ≥ cutoffSim
            std::vector<int> votes(NUM_CLASSES, 0);
            for (auto &pr : sims) {
                if (pr.first < cutoffSim) break;
                votes[pr.second]++;
            }

            // Determine majority vote
            int majorityClass = -1;
            int maxVotes = -1;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                if (votes[c] > maxVotes) {
                    maxVotes = votes[c];
                    majorityClass = c;
                }
            }
            classifications[trueClass][point] = majorityClass;
        }
    }

    // Build confusion matrix
    std::vector<std::vector<long>> confusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));
    for (int trueClass = 0; trueClass < NUM_CLASSES; ++trueClass) {
        for (int p = 0; p < classifications[trueClass].size(); ++p) {
            int predictedClass = classifications[trueClass][p];
            confusionMatrix[trueClass][predictedClass]++;
        }
    }
    return confusionMatrix;
}
