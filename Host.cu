#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "./data_utilities/StatStructs.h"
#include <limits>
#include <future>
#include "./lda/LDA.cpp"
#include <iomanip>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <filesystem>

#include "./cuda_util/CudaUtil.h"
#include "./hyperblock_generation/MergerHyperBlock.cuh"
#include "./hyperblock/HyperBlock.h"
#include "./interval_hyperblock/IntervalHyperBlock.h"
#include "./knn/Knn.h"
#include "./screen_output/PrintingUtil.h"
#include "./data_utilities/DataUtil.h"
#include "./simplifications/Simplifications.h"
using namespace std;

#ifdef _WIN32
    const string PATH_SEPARATOR = "\\";
#else
    const string PATH_SEPARATOR = "/";
#endif

#define LDA_ORDERING true

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset
int COMMAND_LINE_ARGS_CLASS = -1; // used for when we are splitting up generation one class per machine. This lets us run on many computers at once.

map<string, int> CLASS_MAP;
map<string, int> CLASS_MAP_TESTING;

map<int, string> CLASS_MAP_INT;
map<int, string> CLASS_MAP_TESTING_INT;


void evaluateOneToOneHyperBlocks(const std::vector<std::vector<HyperBlock>>& oneToOneHBs,const std::vector<std::vector<std::vector<float>>>& testSet,const std::vector<std::pair<int, int>>& classPairs, int numClasses);


/**
 * For each class of data that we have:
 * Generate "class" and "not class" HBs
 *
 * We will train each class on the opposing points from ALL other classes.
 * For example: Generate HBS for class 0 using the counter points from classes 1,2,3
 *              then generate HBS for (1,2,3) together against counter points from class 0.
 *
 */
vector<vector<HyperBlock>> oneToRestHyper(const vector<vector<vector<float>>>& trainSet, vector<int> ecBestVecIdx) {
    vector<vector<HyperBlock>> oneToRestBlocks;
    vector<HyperBlock> tempHBs;

    const int numClasses = trainSet.size();

    for(int i = 0; i < numClasses; i++) {
        cout << "Training Class (REAL LABEL): " << CLASS_MAP_INT[i] << endl;
        tempHBs.clear();

        // Train with this class as first class, and ALL others as the second class.
        vector<vector<vector<float>>> trainingData(2);
        trainingData[0] = trainSet[i];

        // Add all other class data as the "second class"
        for(int j = 0; j < numClasses; j++) {
            if(j == i) continue;
            for(const auto& point : trainSet[j]) trainingData[1].push_back(point);
        }

        vector<int> bestVecs(2);
        bestVecs[0] = ecBestVecIdx[i];
        bestVecs[1] = ecBestVecIdx[i];

        // Now we generate "HBs for class i" and "HBs for not-class i"
        IntervalHyperBlock::generateHBs(trainingData, tempHBs, bestVecs, FIELD_LENGTH, -1);
        oneToRestBlocks.push_back(tempHBs);

        // Set the HBs generated to have the correct class number.
        for(auto& hb : oneToRestBlocks[i]) {
            if(hb.classNum == 0) {
                hb.classNum = i;                    // Set it to be the correct class index i
                continue;
            }

            hb.classNum = numClasses + i;   // Doing this to be safe in case doing -i for the class would cause indexing issues anywhere.
        }
    }


    return oneToRestBlocks;
}










/**
 * This is the OneToSome based method of HyperBlocks.
 *
 * It will generate hyperblocks for some ordering of the classes.
 * It should be used as a sieve.
 *
    // Based on ordering, each should train on only the ones in front of them.
    // Ex 0, 1, 2, 3, 4, 5
    // 1st round =>  0 : 1, 2, 3, 4, 5
    // 2nd round =>  1 : 2, 3, 4, 5
    // 3rd round =>  2 : 3, 4, 5
 *
 */
vector<vector<HyperBlock>> oneToSomeHyper(const vector<int>& ordering, const vector<vector<vector<float>>>& trainSet, vector<int> ecBestVecIdx) {
    vector<vector<HyperBlock>> oneToSomeBlocks;
    vector<HyperBlock> tempHBs;

    cout << ordering.size() << endl;
    cout << trainSet.size() << endl;
    cout << ecBestVecIdx.size() << endl;

    const int numClasses = trainSet.size();

    for(int i = 0; i < numClasses; i++) {
        cout << "Training Class (REAL LABEL): " << CLASS_MAP_INT[ordering[i]] << "  " << numClasses - i << " more classes." << endl;
        tempHBs.clear();
        // Train with this class as first class, and ALL others (remaining) as the second class.
        vector<vector<vector<float>>> trainingData(2);
        trainingData[0] = trainSet[ordering[i]];

        // Add the training data for each of the remaining classes. (flatten each and put in "point-wise")
        for(int j = i + 1; j < numClasses; j++) {
            for(const auto& point : trainSet[ordering[j]]) {
                trainingData[1].push_back(point);
            }
        }
        vector<int> bestVecs(2);
        bestVecs[0] = ecBestVecIdx[ordering[i]];
        bestVecs[1] = ecBestVecIdx[ordering[i]]; // won't really be used since only gen for first class, so set to same
        int genForClass = 0;

        // Now we should call
        IntervalHyperBlock::generateHBs(trainingData, tempHBs, bestVecs, FIELD_LENGTH, genForClass); // only go to class "0" for generating blocks.
        oneToSomeBlocks.push_back(tempHBs);

        // Set the HBs generated to have the correct class number.
        for(auto& hb : oneToSomeBlocks[i]) {
            hb.classNum = ordering[i];
        }
    }


    return oneToSomeBlocks;
}

/* If we have classes a, b, c. then we should generate blocks to change this to a multi-step 2 class problem
   we will make blocks for a pair at a time.

   a vs b, a vs c, b vs c.

   We will now be able to pass a test set point to be evaluated by each set INDEPENDENTLY.
   We will either treat it as a sieve, ex use the set of blocks that is LEAST over-generalized.
   Future Work: or it can be treated as a voting type of system ex point p gets evaluated on ALL sets
   if it scores a, c, c for the 3 pairs specified above we might want to put it in class c.

   https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
*/
vector<vector<HyperBlock>> oneToOneHyper(const vector<vector<vector<float>>>& trainingData, vector<int> eachClassBestVectorIndex, std::vector<pair<int,int>> &classPairs){
      vector<vector<HyperBlock>> oneToOneHyperBlocks;
      vector<HyperBlock> pairHyperBlocks;
      const int numClasses = trainingData.size();
      cout << "Num Classes" << endl;

    // Make the call to interval hyper FOR EACH set
      for(int i = 0; i < numClasses; i++){
          for(int j = i + 1; j < numClasses; j++){
              pairHyperBlocks.clear();
              cout << i << ", " << j << endl;
              vector<vector<vector<float>>> binaryTrainingData(2);
              binaryTrainingData[0] = trainingData[i];
              binaryTrainingData[1] = trainingData[j];

              // Pass binary training data in . ex class 0 and 1.
              IntervalHyperBlock::generateHBs(binaryTrainingData, pairHyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);

              // Copy all the newly added blocks to the right index in the fin
              oneToOneHyperBlocks.push_back(pairHyperBlocks);
              classPairs.emplace_back(i, j);
              cout << "pDone" << classPairs.size() << endl;
          }
      }

      cout << "Done generating on-to-one HBs, now we are setting their classes to be correct..." << endl;
      // Now what we do.
      for(int i = 0; i < oneToOneHyperBlocks.size(); i++){
         for(int j = 0; j < oneToOneHyperBlocks[i].size(); j++){
            HyperBlock& hb = oneToOneHyperBlocks[i].at(j);

            hb.classNum = (hb.classNum == 0) ? classPairs[i].first : classPairs[i].second;
         }
      }

    return oneToOneHyperBlocks;
}











/**
* We generate a confusion matrix, but allow for points to fall into multiple blocks at a time
* that is why we go through blocks on outerloop and whole dataset on the inside.
*/
float testAccuracyOfHyperBlocks(
    std::vector<HyperBlock>& hyperBlocks,
    std::vector<std::vector<std::vector<float>>>& testSet,
    std::vector<std::vector<std::vector<float>>>& trainingSet,
    std::vector<int> order,
    std::map<std::pair<int, int>, PointSummary>& pointSummaries, // OUTPUT
    int k = 3, float threshold = 0.1
) {
    std::vector<std::unordered_map<int, std::vector<int>>> pointsNotClassified(CLASS_MAP.size());


    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        for (int j = 0; j < testSet[cls].size(); j++) {
            pointsNotClassified[cls][j] = std::vector<int>(NUM_CLASSES);
        }
    }

    for (int hb = 0; hb < hyperBlocks.size(); hb++) {
        HyperBlock& currBlock = hyperBlocks[hb];
        for (int cls = 0; cls < NUM_CLASSES; cls++) {
            for (int pnt = 0; pnt < testSet[cls].size(); pnt++) {
                const std::vector<float>& point = testSet[cls][pnt];
                if (currBlock.inside_HB(point.size(), point.data())) {
                    pointsNotClassified[cls][pnt][currBlock.classNum]++;

                    // Make the point summary entry for this point, might want to double check this thing.
                    PointSummary& summary = pointSummaries[std::make_pair(cls, pnt)];
                    summary.classIdx = cls;
                    summary.pointIdx = pnt;

                    // make the blockinfo
                    BlockInfo info;
                    info.blockClass = currBlock.classNum;
                    info.blockIdx = hb;
                    info.blockSize = currBlock.size;
                    info.blockDensity = -1;

                    summary.blockHits.push_back(info);
                }
            }
        }
    }

    // Assign predicted class
    std::map<std::pair<int, int>, PointSummary>::iterator it;
    for (it = pointSummaries.begin(); it != pointSummaries.end(); ++it) {
        PointSummary& summary = it->second;
        std::map<int, int> classVotes;
        for (size_t i = 0; i < summary.blockHits.size(); ++i) {
            classVotes[summary.blockHits[i].blockClass]++;
        }
        int maxVotes = -1;
        int predictedClass = -1;
        std::map<int, int>::iterator voteIt;
        for (voteIt = classVotes.begin(); voteIt != classVotes.end(); ++voteIt) {
            if (voteIt->second > maxVotes) {
                maxVotes = voteIt->second;
                predictedClass = voteIt->first;
            }
        }
        summary.predictedIdx = predictedClass;
    }

    //TODO: PUT IN THE KNN HERE,
    //TODO  add to PointSummary with info from KNN fallback pls

    
  /*
    // Find the closest actual cases to the edge of each block.
    //for(HyperBlock& hb: hyperBlocks){
        //hb.tameBounds(trainingSet);
    //}

    // this is where we have lots of options. this graveyard is all the different "unclassified classifiers" we have tried.
    //std::vector<std::vector<long>> secondConfusionMatrix = Knn::closeToInkNN(unclassifiedPointVec, hyperBlocks, k, NUM_CLASSES);
    // std::vector<std::vector<long>> secondConfusionMatrix = Knn::closestBlock(unclassifiedPointVec, hyperBlocks, NUM_CLASSES);
    //std::vector<std::vector<long>> secondConfusionMatrix = Knn::pureKnn(unclassifiedPointVec, trainingSet,  k, NUM_CLASSES);
    //std::vector<std::vector<long>> secondConfusionMatrix = Knn::bruteMergable(unclassifiedPointVec, trainingSet, hyperBlocks, 5, NUM_CLASSES);
    // vector<vector<long>> secondConfusionMatrix = Knn::closeToInkNN(unclassifiedPointVec, hyperBlocks, k, NUM_CLASSES);
    // vector<vector<long>> secondConfusionMatrix = Knn::kNN(unclassifiedPointVec, hyperBlocks, k, NUM_CLASSES);
    // vector<vector<long>> secondConfusionMatrix = Knn::mergableKNN(unclassifiedPointVec, trainingSet, hyperBlocks, NUM_CLASSES);
    vector<vector<long>> secondConfusionMatrix = Knn::thresholdKNN(unclassifiedPointVec, trainingSet, NUM_CLASSES, k, threshold);
    PrintingUtil::printConfusionMatrix(secondConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    cout << "============================ END K-NN MATRIX ======================" << endl;

    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            regularConfusionMatrix[i][j] = regularConfusionMatrix[i][j] + secondConfusionMatrix[i][j];
        }
    }
    */


    return 0.0f; // placeholder
}





/* This function computes the LDA ordering for a given training dataset.
 * It sets up the bestVectors, bestVectorsIndexes, and eachClassBestVectorIndex.
 * best vectors is the weights of each coefficient from the LDF function
 * bestVectorsIndexes is just the indexes that correspond to those weights from the function, since we are sorting them
 * eachClassBestVectorIndex is the one best attribute for each class, we sort by this when generating blocks, and it helps a bit.
*
 * returns the class accuracy ordering so we can use the "sift approach" for HBs ex {0, 4, 2, 1, 5, 6, 7, 8, 9} for MNIST classes 0 is best seperable, 9 would be worst.
 */
std::vector<int> computeLDAOrdering(const vector<vector<vector<float>>>& trainingData, vector<vector<float>>& bestVectors, vector<vector<int>>& bestVectorsIndexes, vector<int>& eachClassBestVectorIndex) {
    // Run LDA on the training data.
    std::pair<std::vector<std::vector<float>>, std::vector<int>> result = linearDiscriminantAnalysis(trainingData);
    std::vector<int> classOrder;
    bestVectors = result.first;

    // Accuracy ordering of the LDA on the training data.
    classOrder = result.second;

    // Resize our index containers.
    bestVectorsIndexes.assign(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
    eachClassBestVectorIndex.assign(NUM_CLASSES, 0);

    // For each class, initialize the indexes and then sort (if desired)
    // and determine the index with the largest absolute LDA coefficient.
    for (int i = 0; i < NUM_CLASSES; i++) {
        // Populate with initial indices: 0, 1, 2, ... FIELD_LENGTH - 1.
        for (int j = 0; j < FIELD_LENGTH; j++) {
            bestVectorsIndexes[i][j] = j;
        }

#ifdef LDA_ORDERING
        // Optionally sort the indexes for class i based on the absolute value of the LDA coefficients.
        sort(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
             [&](int a, int b) {
                 return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
             });
#endif
        // Find the index (from bestVectorsIndexes) corresponding to the largest absolute LDA coefficient.
        // We use the values in bestVectors[i] for comparison.
        auto it = max_element(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
                              [&](int a, int b) {
                                  return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
                              });
        eachClassBestVectorIndex[i] = distance(bestVectorsIndexes[i].begin(), it);
    }

    return classOrder;
}


vector<float> runKFold(vector<vector<vector<float>>> &dataset, std::vector<std::pair<int,int>>& classPairs, bool onToMany = true, bool takeUserInput = false, int removalCount = 3, int nearestNeighborK = 5, float similarityThreshold = 0.6f, bool hidePrinting = false) {

    if (dataset.empty()) {
        cout << "Please enter a training dataset before using K Fold validation" << endl;
        return {-1, -1, -1};
    }

    int k;
    // if we're taking input run it like normal. using this variable lets us just do it this way.
    if (takeUserInput) {
        cout << "Please Enter a K value:\t";
        cin >> k;

        // Clear the newline from the input buffer.
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        if (cin.fail() || k < 2) {
            cout << "Error: Invalid input. Please enter a valid integer greater than 1." << endl;
            // Clear the error state and ignore any remaining input.
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            return {-1, -1, -1};
        }
    }
    // if we're not using user input, we are testing for best accuracy and we can use 10.
    else
        k = 10;


    // used to hide the printing of the regular kFold testing stuff. so that when we are finding best parameters we don't have all that printing
    streambuf* oldBuf = nullptr;
    ostringstream nullSink;          // collects nothing, never flushed
    if (hidePrinting) {
        oldBuf = cout.rdbuf(nullSink.rdbuf());   // silence everything
    }

    vector<vector<vector<vector<float>>>> kFolds = DataUtil::splitDataset(dataset, k);
    // stats trackers for cross folds.
    float acc = 0.0f;
    int blockCount = 0;
    int cCount = 0;

    // generate blocks with a training set which is all folds except i. using i as the test dataset.
    for (int i = 0; i < k; i++) {
        // trainingData will store all folds except the i-th as training data.
        vector<vector<vector<float>>> trainingData(NUM_CLASSES);

        // Loop through all folds except i and accumulate points by class.
        for (int fold = 0; fold < k; fold++) {
            if (fold == i) continue; // skip test fold

            // build our training data
            for (int cls = 0; cls < NUM_CLASSES; cls++) {
                // Append all points from kFolds[fold][cls] to trainingData[cls]
                trainingData[cls].insert(trainingData[cls].end(), kFolds[fold][cls].begin(), kFolds[fold][cls].end());
            }

        }

        // The test dataset for this iteration is simply fold i.
        vector<vector<vector<float>>> testData = kFolds[i];

        // now that our data is set up with training and testing, we simply do business as usual. we are going to do our LDA on the train data, then just do our block generation and simplification
        // Run LDA on the training data.
        vector<vector<float>>bestVectors;

        // Initialize indexes for each class
        vector<vector<int>> bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
        vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);

        computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);


        cout << "----------------------------FOLD " << (i + 1) << " RESULTS----------------------------------" << endl;

        if (oneToMany) {
            // ------------------------------------------
            // GENERATING BLOCKS BUSINESS AS USUAL
            vector<HyperBlock> hyperBlocks;

            IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
            //cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
            //cout << "GENERATED WITH " << hyperBlocks.size() << " BLOCKS" << endl;
            vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);


        // make our blocks
        IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);

        // simplify them, with the simplification count we have specifed as a parameter. usually 0, but playing with this value can get us better results because we are removing more blocks
        Simplifications::REMOVAL_COUNT = removalCount;

        // the vector of int is the total times it simplified, then the clause count.
        vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);

        // clause count computed here because sometimes we don't simplify
        int totalPoints = 0;
        for (const auto &c : trainingData)
            totalPoints += c.size();

        int clauseCount = 0;
        for (const auto &hb : hyperBlocks) {
            for (int a = 0; a < FIELD_LENGTH; a++) {
                if (hb.minimums[a][0] != 0.0f || hb.maximums[a][0] != 1.0f)
                    clauseCount++;
                }
            }

        } // end of one train/test loop
        else {
            vector<vector<HyperBlock>> oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex, classPairs);
            evaluateOneToOneHyperBlocks(oneToOneBlocks, testData, classPairs, NUM_CLASSES);
        }

    }

        acc += testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData, nearestNeighborK, similarityThreshold);
        blockCount += hyperBlocks.size();
        cCount += clauseCount;
    } // end of one train/test loop

    float avgAcc = float (acc) / float(k);
    float blockAvg = float(blockCount) / float(k);
    float clauseAvg = float(cCount) / float(k);

    if (hidePrinting) cout.rdbuf(oldBuf);           // back to console
        
    cout << "OVERALL ACCURACY " << avgAcc << endl;
    cout << "Average block count " << blockAvg << endl;
    cout << "Average clause count " << clauseAvg << endl;

    return {avgAcc, blockAvg, clauseAvg};
}

// function which is going to test brute force with different removing useless blocks thresholds and k values for KNN to determine best accuracy we can find across k fold validation with 10 folds.
void findBestParameters(vector<vector<vector<float>>> &dataset, int maxRemovalCount, int maxK) {

    if (dataset.empty()) {
        cout << "Please enter a training dataset before testing parameters" << endl;
        return;
    }

    vector<float> bestResults = {-1, -1, -1};
    vector<int> bestParameters = {-1, -1, -1};
    for (int removalCount = 0; removalCount < maxRemovalCount; removalCount++) {
        for (int k = 1; k <= maxK; k += 2) {
            for (int threshold = 1; threshold < 11; threshold++){
                float t = 0.1f * (float)threshold;
                vector<float> results = runKFold(dataset, false, removalCount, k, t, true);
                if (results[0] > bestResults[0]) {
                    bestResults = results;
                    bestParameters = {removalCount, k, threshold};
                }
                cout << "Removal Count: " << removalCount << " K: " << k << " Threshold: " << t <<endl;
            }
        }
    }
    cout << "Best accuracy: " << bestResults[0] << "\nBlock Count Average w/best results: " << bestResults[1] << "\nClause Count Average: " << bestResults[2] << endl;
    cout << "BEST PARAMETERS: REMOVAL COUNT: " << bestParameters[0] << " K: " << bestParameters[1] << " Threshold: " << bestParameters[2] << endl;
}


float evaluateOneToSomeHBs(const vector<vector<HyperBlock>>& oneToSomeBlocks, const vector<vector<vector<float>>>& testData) {
    vector<vector<long>> confusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));
    int totalPoints = 0;
    for(const auto& c : testData) {
        totalPoints += c.size();
    }

    int incorrect = 0;
    int correct = 0;
    int pointsTested = 0;
    // Go through the classes
    for(int i = 0; i < NUM_CLASSES; i++) {
        for(int j = 0; j < testData[i].size(); j++) {
            pointsTested++;
            const auto& point = testData[i][j];
            bool classified = false;

            for(int sieveLvl = 0; sieveLvl < oneToSomeBlocks.size(); sieveLvl++) {
                for(const auto& hb : oneToSomeBlocks[sieveLvl]) {
                    if(hb.inside_HB(point.size(), point.data())) {
                        classified = true;

                        // If it is of the wrong class.
                        if(hb.classNum != i) {
                            incorrect++;
                        }
                        else {
                            correct++;
                        }

                        confusionMatrix[i][hb.classNum]++;
                        break;
                    }
                }
                if (classified) break; // do not keep checking once classified
            }
        }
    }

    for (const auto& row : confusionMatrix) {
        for (int i = 0; i < row.size(); i++) {
            std::cout << std::setw(5) << row[i] << " ";
        }
        std::cout << std::endl;
    }

    PrintingUtil::printConfusionMatrix(confusionMatrix, NUM_CLASSES, CLASS_MAP_INT);

    // Calculate the coverage.
    float coverage = static_cast<float>(incorrect + correct) / static_cast<float>(totalPoints);
    float acc = static_cast<float>(correct) / static_cast<float>(correct + incorrect);

    cout << "Coverage %: " << coverage * 100.0f << "%" << endl;
    cout << "Coverage total: " << (incorrect + correct) << " out of " << totalPoints << endl;
    cout << "Overall Accuracy: " << acc * 100.0f << "%" << endl;
    cout << "Total number of points tested: " << pointsTested << endl;
    return acc;
}



// -------------------------------------------------------------------------
// Asynchronous mode: run when argc >= 2
int runAsync(int argc, char* argv[]) {
    // Local variables for async mode
    string normalizedSaveFile;
    string hyperBlocksImportFileName;
    string trainingDataFileName;
    string testingDataFileName;
    string hyperBlocksExportFileName;

    // 3-D DATASETS
    vector<vector<vector<float>>> testData;
    vector<vector<vector<float>>> trainingData;

    // Normalization vectors (will be resized later)
    vector<float> minValues;
    vector<float> maxValues;

    // Store our HyperBlocks
    vector<HyperBlock> hyperBlocks;

    // Ultra confusion matrix
    vector<vector<long>> ultraConfusionMatrix;

    // Variables to be set by LDA

    if (argc > 3) {
        cout << "TOO MANY ARGUMENTS!" << endl;
        exit(1);
    }

    if (argc == 3) {
        // Set a global or externally-declared variable
        COMMAND_LINE_ARGS_CLASS = stoi(argv[2]);
        cout << "Running on class index " << COMMAND_LINE_ARGS_CLASS << endl;
    }

    // Process training data from file provided as first argument
    trainingData = DataUtil::dataSetup(argv[1], CLASS_MAP, CLASS_MAP_INT);
    cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << endl;
    cout << "NUM CLASSES : " << NUM_CLASSES << endl;

    // Resize normalization vectors based on FIELD_LENGTH
    minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
    maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());

    DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
    DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

    cout << "RUNNING LDA" << endl;
    // Run LDA on the training data.
    vector<vector<float>>bestVectors;
    // Initialize indexes for each class
    vector<vector<int>> bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
    vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);
    computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);

    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
    cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
    cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << endl;
    cout << "BEGINNING SIMPLIFICATIONS" << endl;

    string nonSimplified = string("NonSimplifiedBlocksClass") + to_string(COMMAND_LINE_ARGS_CLASS);
    DataUtil::saveBasicHBsToCSV(hyperBlocks, nonSimplified, FIELD_LENGTH);

    vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
    int totalPoints = 0;
    for (const auto &c : trainingData)
        totalPoints += c.size();
    cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
    cout << "Ran simplifications: " << result[0] << " Times" << endl;
    cout << "We had: " << totalPoints << " points\n";

    string simplified = string("SimplifiedBlocks") + to_string(COMMAND_LINE_ARGS_CLASS);
    DataUtil::saveBasicHBsToCSV(hyperBlocks, simplified, FIELD_LENGTH);
    return 0;
}


void evaluateOneToOneHyperBlocks(
    const std::vector<std::vector<HyperBlock>>& oneToOneHBs,
    const std::vector<std::vector<std::vector<float>>>& testSet,
    const std::vector<std::pair<int, int>>& classPairs,
    int numClasses
) {
    int totalPoints = 0;
    int correctPoints = 0;

    std::vector<std::vector<int>> confusion(numClasses, std::vector<int>(numClasses, 0));
    std::vector<int> correctPerClass(numClasses, 0);
    std::vector<int> totalPerClass(numClasses, 0);


    for (int actualClass = 0; actualClass < numClasses; ++actualClass) {
        for (const auto& point : testSet[actualClass]) {
            std::vector<int> votes(numClasses, 0);

            // Run all pairwise comparisons
            for (int i = 0; i < numClasses; ++i) {
                for (int j = i + 1; j < numClasses; ++j) {
                    // Find the correct index using classPairs
                    int findPairIndex = -1;
                    for (int idx = 0; idx < classPairs.size(); ++idx) {
                        if ((classPairs[idx].first == i && classPairs[idx].second == j) ||
                            (classPairs[idx].first == j && classPairs[idx].second == i)) {
                            findPairIndex = idx;
                            break;
                            }
                    }
                    if (findPairIndex == -1) {
                        std::cerr << "Error: could not find block set for classes " << i << " and " << j << "\n";
                        continue;
                    }

                    const auto& pairHBs = oneToOneHBs[findPairIndex];


                    for (const auto& block : pairHBs) {
                        if (block.classNum == i && block.inside_HB(point.size(), point.data())) {
                            votes[i]++;
                        }
                        if (block.classNum == j && block.inside_HB(point.size(), point.data())) {
                            votes[j++];
                        }
                    }
                }
            }

            // Majority vote
            int predictedClass = std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
            if (votes[predictedClass] == 0) {
                // No vote: optionally handle as unclassified
                continue;
            }

            confusion[actualClass][predictedClass]++;
            totalPerClass[actualClass]++;
            if (predictedClass == actualClass) {
                correctPerClass[actualClass]++;
                correctPoints++;
            }
            totalPoints++;
        }
    }

    // Output
    std::cout << "\nConfusion Matrix:\n";
    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            std::cout << confusion[i][j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nPer-Class Accuracy:\n";
    for (int i = 0; i < numClasses; ++i) {
        float acc = totalPerClass[i] ? static_cast<float>(correctPerClass[i]) / totalPerClass[i] : 0.0f;
        std::cout << "Class " << i << ": " << acc * 100.0f << "%\n";
    }

    std::cout << "\nOverall Accuracy: " << (static_cast<float>(correctPoints) / totalPoints) * 100.0f << "%\n";
}

/**
 *  We will attempt to find some "best" ordering of the classes by training normal HBs, then calling this function,
 *  this function will return the order that we believe is best to be used by the one-to-some method.
 *
 *  This is similar to using the LDA accuracies to sort the classes as best -> worst accuracy, but
 *  it is closer to what we should actually expect our blocks to do. The LDA is inherently different from
 *  how our blocks classify, and thus this should hopefully give us a better order.
 *
 *  Order will be determined by seeing which classes HyperBlocks swallow the most points from the wrong classes.
 *  (we also keep track of how many times a point of a class falls into wrong blocks)
 *
 * @param validationData The validation data set.
 * @param hyperBlocks The HyperBlocks which were trained WITHOUT using the validation points.
 * @return a vector<int> which is the recommended order to create one-to-some blocks in.
 */
vector<int> findOneToSomeOrder(vector<vector<vector<float>>>& validationData, vector<HyperBlock>& hyperBlocks) {
    int numClasses = validationData.size();

    // Keep track of how many times a block from each class picks up points from other classes.
    vector<int> hbClassOverclaims(numClasses, 0);  // based on the hyperblock.class
    // This one will keep track of how many times A POINT from a class falls into multiple blocks.
    vector<vector<int>> sneakyPointCount(numClasses, vector<int>(numClasses, 0));

    for(int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < validationData[i].size(); ++j) {
            const auto& point = validationData[i][j];

            for(const auto& block : hyperBlocks) {
                if(block.inside_HB(point.size(), point.data())) {
                    if(i != block.classNum) { // Incorrect classification.
                        hbClassOverclaims[block.classNum]++;    // How many times HBs classify the WRONG point!
                        sneakyPointCount[i][block.classNum]++;  ;  // How many times POINTS of a class fall into wrong class HBs
                    }
                }
            }
        }
    }

    // Now we should look at the numbers for hbClassOverclaims to analyze which ordering would be the best.
    // Save raw counts (temps)
    std::ofstream outFile("hbClassOverclaims.csv"); if (!outFile.is_open()) {std::cerr << "Error opening file for writing: " << "hbClassOverclaims.csv" << std::endl;}for (size_t i = 0; i < hbClassOverclaims.size(); ++i) {outFile << i << "," << hbClassOverclaims[i] << "\n";}
    std::ofstream outFile2("sneakyPointCount.csv");if (!outFile2.is_open()) {std::cerr << "Error opening file for writing: " << "sneakyPointCount.csv" << std::endl;}for (size_t i = 0; i < sneakyPointCount.size(); ++i) {for (size_t j = 0; j < sneakyPointCount[i].size(); ++j) {outFile2 << sneakyPointCount[i][j];if (j + 1 != sneakyPointCount[i].size())outFile2 << ",";}outFile2 << "\n";}


    // Build mistake counts
    vector<std::pair<int, int>> pointMistakes;   // (count, class)
    vector<std::pair<int, int>> blockMistakes;   // (count, class)
    vector<std::pair<int, int>> combinedMistakes; // (count, class)

    for (int c = 0; c < numClasses; ++c) {
        int pointMistakeSum = 0;
        for (int other = 0; other < numClasses; ++other) {
            pointMistakeSum += sneakyPointCount[c][other];
        }

        int blockMistake = hbClassOverclaims[c];
        int combined = pointMistakeSum + blockMistake;

        pointMistakes.emplace_back(pointMistakeSum, c);
        blockMistakes.emplace_back(blockMistake, c);
        combinedMistakes.emplace_back(combined, c);
    }

    // Now sort each list
    auto sorter = [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first < b.first; // sort by mistake counts ascending
    };

    std::sort(pointMistakes.begin(), pointMistakes.end(), sorter);
    std::sort(blockMistakes.begin(), blockMistakes.end(), sorter);
    std::sort(combinedMistakes.begin(), combinedMistakes.end(), sorter);

    // Extract orders
    vector<int> pointOrder, blockOrder, combinedOrder;
    for (auto& p : pointMistakes) pointOrder.push_back(p.second);
    for (auto& p : blockMistakes) blockOrder.push_back(p.second);
    for (auto& p : combinedMistakes) combinedOrder.push_back(p.second);

    // Save the ordering. (temp)
    std::ofstream orderFile("orderings.csv"); if (!orderFile.is_open()) { std::cerr << "Error opening file for writing orderings.csv" << std::endl; } else { orderFile << "pointOrder:"; for (size_t i = 0; i < pointOrder.size(); ++i) { orderFile << " " << pointOrder[i]; if (i + 1 != pointOrder.size()) orderFile << ","; } orderFile << "\n"; orderFile << "blockOrder:"; for (size_t i = 0; i < blockOrder.size(); ++i) { orderFile << " " << blockOrder[i]; if (i + 1 != blockOrder.size()) orderFile << ","; } orderFile << "\n"; orderFile << "combinedOrder:"; for (size_t i = 0; i < combinedOrder.size(); ++i) { orderFile << " " << combinedOrder[i]; if (i + 1 != combinedOrder.size()) orderFile << ","; } orderFile << "\n"; orderFile.close(); }

    return blockOrder;
}


// -------------------------------------------------------------------------
// Interactive mode: run when argc < 2
void runInteractive() {
    // Local variables for interactive mode
    string normalizedSaveFile;
    string hyperBlocksImportFileName;
    string trainingDataFileName;
    string testingDataFileName;
    string hyperBlocksExportFileName;

    vector<vector<vector<float>>> testData;
    vector<vector<vector<float>>> trainingData;
    vector<vector<vector<float>>> validationData;

    vector<float> minValues;
    vector<float> maxValues;

    vector<HyperBlock> hyperBlocks;

    vector<vector<long>> ultraConfusionMatrix;

    vector<vector<float>> bestVectors;
    vector<vector<int>> bestVectorsIndexes;
    vector<int> eachClassBestVectorIndex;

    // Class ordering
    vector<int> order;
    vector<vector<HyperBlock>> oneToOneBlocks;
    std::vector<std::pair<int, int>> classPairsOut;

    vector<vector<HyperBlock>> oneToRestBlocks;

    int normChoice;

    bool running = true;
    int choice;
    while (running) {
        PrintingUtil::displayMainMenu();
        cin >> choice;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        switch (choice) {
            case 1: { // IMPORT TRAINING DATA
                cout << "Enter training data filename: " << endl;
                #ifdef _WIN32
                    system("dir DATASETS");
                #else
                    system("ls DATASETS");
                #endif
                getline(cin, trainingDataFileName);
                string fullPath = "DATASETS" + string(PATH_SEPARATOR) + trainingDataFileName;
                CLASS_MAP.clear();
                CLASS_MAP_INT.clear();
                trainingData = DataUtil::dataSetup(fullPath.c_str(), CLASS_MAP, CLASS_MAP_INT);

                cout << "Choose normalization method:\n";
                cout << "  1. Min-Max normalize using dataset bounds\n";
                cout << "  2. Normalize by fixed max value (e.g., 255)\n";
                cout << "  3. No normalization\n";
                cout << "Enter choice (1-3): ";
                cin >> normChoice;
                cin.ignore();  // flush newline


                if (normChoice == 1) {
                    minValues.assign(FIELD_LENGTH, std::numeric_limits<float>::infinity());
                    maxValues.assign(FIELD_LENGTH, -std::numeric_limits<float>::infinity());
                    DataUtil::findMinMaxValuesInDataset(trainingData, minValues, maxValues, FIELD_LENGTH);
                    DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);

                    order = computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);
                } else if (normChoice == 2) {
                    float fixedMax;
                    cout << "Enter fixed max value (e.g., 255): ";
                    cin >> fixedMax;
                    cin.ignore();  // flush newline

                    minValues.assign(FIELD_LENGTH, 0.0f);
                    maxValues.assign(FIELD_LENGTH, fixedMax);

                    DataUtil::minMaxNormalization(trainingData, minValues, maxValues, FIELD_LENGTH);
                    order = computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);
                } else {
                    cout << "Skipping normalization.\n";
                }


                PrintingUtil::waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                cout << "Enter testing data filename: " << endl;
                system("ls DATASETS");
                getline(cin, testingDataFileName);
                string fullPath = "DATASETS" + string(PATH_SEPARATOR) + testingDataFileName;

                // clear these two maps to prevent issues when using a second test set.
                CLASS_MAP_TESTING.clear();
                CLASS_MAP_TESTING_INT.clear();

                testData = DataUtil::dataSetup(fullPath, CLASS_MAP_TESTING, CLASS_MAP_TESTING_INT);

                if (normChoice == 1 || normChoice == 2) {
                    DataUtil::normalizeTestSet(testData, minValues, maxValues, FIELD_LENGTH);
                } else {
                    cout << "Skipping normalization.\n";
                }

                // Normalize and reorder testing data as needed.
                testData = DataUtil::reorderTestingDataset(testData, CLASS_MAP, CLASS_MAP_TESTING);

                for(const auto& cls: testData) {
                    cout << cls.size() << endl;
                }
                PrintingUtil::waitForEnter();
                break;
            }
            case 3: { // SAVE NORMALIZED TRAINING DATA
                cout << "Enter the file to save the normalized training data to: " << endl;
                getline(cin, normalizedSaveFile);
                DataUtil::saveNormalizedVersionToCsv(normalizedSaveFile, trainingData);
                cout << "Saved normalized training data to: " << normalizedSaveFile << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 4: { // IMPORT EXISTING HYPERBLOCKS
                cout << "Enter existing hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);
                hyperBlocks = DataUtil::loadBasicHBsFromBinary(hyperBlocksImportFileName);
                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;

				//try_expand_blocks_and_save(hyperBlocks, trainingData);

                for(HyperBlock& hb: hyperBlocks){
                  hb.find_avg_and_size(trainingData);
                }

                PrintingUtil::waitForEnter();
                break;
            }
            case 5: { // EXPORT HYPERBLOCKS
                cout << "Enter the file to save HyperBlocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                DataUtil::saveBasicHBsToCSV(hyperBlocks, hyperBlocksExportFileName, FIELD_LENGTH);
                break;
            }
            case 6: { // GENERATE NEW HYPERBLOCKS
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    PrintingUtil::waitForEnter();
                } else {
                    hyperBlocks.clear();
                    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
                }

                cout << "Finished Generating HyperBlocks" << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 7: {
                // SIMPLIFY HYPERBLOCKS
                vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
                int totalPoints = 0;

                for (const auto &c : trainingData) totalPoints += c.size();

                //for(HyperBlock& hb: hyperBlocks) hb.findSize(trainingData);

                cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
                cout << "We got a final total of: " << hyperBlocks.size() << " blocks." << endl;
                cout << "We had: " << totalPoints << " points of training data\n";
                PrintingUtil::waitForEnter();
                break;
            }
            case 8: { // TEST HYPERBLOCKS ON DATASET
                std::cout << "Testing hyperblocks on testing dataset" << std::endl;
                testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData, order);

                PrintingUtil::waitForEnter();
                break;
            }
            case 9: {
                evaluateOneToOneHyperBlocks(oneToOneBlocks, testData, classPairsOut, NUM_CLASSES);
                PrintingUtil::waitForEnter();
                break;
            }
            case 10: {
                runKFold(trainingData, true, classPairsOut);
                PrintingUtil::waitForEnter();
                break;
            }
            case 11: {
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    PrintingUtil::waitForEnter();
                } else {
                    oneToOneBlocks.clear();
                    oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex, classPairsOut);
                }

                cout << "Finished Generating 1-1 HyperBlocks" << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 12: {
                //      oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex);

                // Import 1-1 Hyperblocks
                cout << "Enter 1-1 Hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);
                oneToOneBlocks = DataUtil::loadOneToOneHBsFromBinary(hyperBlocksImportFileName, classPairsOut);
                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;

                /* Might still want this but has to be wrapped in loop through each pair.
                for(HyperBlock& hb: hyperBlocks){
                    hb.find_avg_and_size(trainingData);
                }

                blockSizeDistribution(hyperBlocks);
                */

                PrintingUtil::waitForEnter();
                break;
            }
            case 13: {
                // Export 1-1 Hyperblocks
                cout << "Enter the file to save HyperBlocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                DataUtil::saveOneToOneHBsToBinary(oneToOneBlocks, hyperBlocksExportFileName, FIELD_LENGTH);
                break;
            }
            case 14: {
                // RUN K FOLD FOR THE ONE TO ONE TYPE BLOCKS.
                runKFold(trainingData, false, classPairsOut);
                PrintingUtil::waitForEnter();
                break;
            }
            case 15: {

                // POINT BASED ORDER
                oneToRestBlocks.clear();
                // Train the oneToSome Blocks
                auto start = std::chrono::high_resolution_clock::now();
                oneToRestBlocks = oneToRestHyper(trainingData, eachClassBestVectorIndex);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end - start;

                // Flatten the set of HBs by moving them.
                vector<HyperBlock> allRestBlocks;
                for (auto& blockSet : oneToRestBlocks) {
                    allRestBlocks.insert(allRestBlocks.end(),std::make_move_iterator(blockSet.begin()),std::make_move_iterator(blockSet.end()));
                }

                DataUtil::saveBasicHBsToBinary(allRestBlocks, "digitBlocksRest.csv", FIELD_LENGTH);
                cout << "Finished Generating one to Some blocks." << endl;
                std::cout << "Elapsed time: " << diff.count() << " seconds\n";
            }
            case 16: { 
              
                int maxRemoval;
                int maxK;

                cout << "Please enter a max threshold size to test our blocks with " << endl;
                cin >> maxRemoval;
                // Clear the newline from the input buffer.
                cin.ignore(numeric_limits<streamsize>::max(), '\n');

                if (cin.fail() || maxRemoval < 0) {
                    cout << "Error: Invalid input. Please enter a valid integer greater than 1." << endl;
                    // Clear the error state and ignore any remaining input.
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    return;
                }

                cout << "Please enter a max K value to test our KNN with " << endl;
                cin >> maxK;
                // Clear the newline from the input buffer.
                cin.ignore(numeric_limits<streamsize>::max(), '\n');

                if (cin.fail() || maxK < 0) {
                    cout << "Error: Invalid input. Please enter a valid integer greater than 1." << endl;
                    // Clear the error state and ignore any remaining input.
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    return;
                }

                findBestParameters(trainingData, maxRemoval, maxK);
                PrintingUtil::waitForEnter();
                break;
            }
            default: {
                cout << "\nInvalid choice. Please try again." << endl;
                PrintingUtil::waitForEnter();
                break;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Main entry point: choose mode based on argc.
int main(int argc, char* argv[]) {

    // Command line input mode, allows you to specify in command line what to do
    if (argc >= 2)
        return runAsync(argc, argv);

    // Interactive input loop, options to import data, train, test, save, etc
    runInteractive();
    return 0;
}