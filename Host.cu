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
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <tuple>
#include "./hyperblock_generation/MergerHyperBlock.cuh"
#include "./hyperblock/HyperBlock.h"
#include "./interval_hyperblock/IntervalHyperBlock.h"
#include "./knn/Knn.h"
#include "./screen_output/PrintingUtil.h"
#include "./data_utilities/DataUtil.h"
#include "./simplifications/Simplifications.h"
#include "classification_testing/ClassificationTests.h"
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

// USED SO THAT WE CAN GET THE NAMES OF EACH CLASS BASICALLY FOR TRAIN AND TEST DATA
map<string, int> CLASS_MAP;
map<string, int> CLASS_MAP_TESTING;
map<int, string> CLASS_MAP_INT;
map<int, string> CLASS_MAP_TESTING_INT;

void evaluateOneToOneHyperBlocks(const vector<vector<HyperBlock>>& oneToOneHBs,const vector<vector<vector<float>>>& testSet,const vector<pair<int, int>>& classPairs, int numClasses);

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


/* If we have classes a, b, c. then we should generate blocks to change this to a multi-step 2 class problem
   we will make blocks for a pair at a time.

   a vs b, a vs c, b vs c.

   We will now be able to pass a test set point to be evaluated by each set INDEPENDENTLY.
   We will either treat it as a sieve, ex use the set of blocks that is LEAST over-generalized.
   Future Work: or it can be treated as a voting type of system ex point p gets evaluated on ALL sets
   if it scores a, c, c for the 3 pairs specified above we might want to put it in class c.

   https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
*/
vector<vector<HyperBlock>> oneToOneHyper(const vector<vector<vector<float>>>& trainingData, vector<int> eachClassBestVectorIndex, vector<pair<int,int>> &classPairs){
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

float testAccuracyOfHyperBlocks(vector<HyperBlock> &hyperBlocks, vector<vector<vector<float>>> &testData, vector<vector<vector<float>>> &trainingData, map<pair<int, int>, PointSummary>& pointSummaries, int k = 5, float threshold = 0.25) {

    // get our confusion matrix by just classifying with the blocks like normal
    vector<vector<vector<float>>> notClassifiedPoints(NUM_CLASSES);
    vector<vector<long>> hyperBlocksConfusionMatrix = ClassificationTests::buildConfusionMatrix(hyperBlocks, trainingData, testData, ClassificationTests::HYPERBLOCKS, notClassifiedPoints, NUM_CLASSES, pointSummaries);

    cout << "------------------------HYPERBLOCKS CONFUSION MATRIX-----------------------------" << endl;
    float hbAccuracy = PrintingUtil::printConfusionMatrix(hyperBlocksConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);

    // now build our second confusion matrix out of the unclassified stuff only
    vector<vector<vector<float>>> stillNotClassifiedPoints(NUM_CLASSES);
    vector<vector<long>> knnMatrix = ClassificationTests::buildConfusionMatrix(hyperBlocks, trainingData, notClassifiedPoints, ClassificationTests::PURE_KNN, stillNotClassifiedPoints, NUM_CLASSES, pointSummaries, k, threshold);

    cout << "------------------------KNN CONFUSION MATRIX--------------------------------" << endl;
    float knnAccuracy = PrintingUtil::printConfusionMatrix(knnMatrix, NUM_CLASSES, CLASS_MAP_INT);

    vector<vector<long>> finalConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));
    // now we can just combine the two matrices
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            finalConfusionMatrix[i][j] = knnMatrix[i][j] + hyperBlocksConfusionMatrix[i][j];
        }
    }

    cout << "-------------------------FINAL (HB + KNN predictions) CONFUSION MATRIX--------------------------" << endl;
    float finalAccuracy = PrintingUtil::printConfusionMatrix(finalConfusionMatrix, NUM_CLASSES, CLASS_MAP_INT);
    return finalAccuracy;
}

/* This function computes the LDA ordering for a given training dataset.
 * It sets up the bestVectors, bestVectorsIndexes, and eachClassBestVectorIndex.
 * best vectors is the weights of each coefficient from the LDF function
 * bestVectorsIndexes is just the indexes that correspond to those weights from the function, since we are sorting them
 * eachClassBestVectorIndex is the one best attribute for each class, we sort by this when generating blocks, and it helps a bit.
*
 * returns the class accuracy ordering so we can use the "sift approach" for HBs ex {0, 4, 2, 1, 5, 6, 7, 8, 9} for MNIST classes 0 is best seperable, 9 would be worst.
 */
vector<int> computeLDAOrdering(const vector<vector<vector<float>>>& trainingData, vector<vector<float>>& bestVectors, vector<vector<int>>& bestVectorsIndexes, vector<int>& eachClassBestVectorIndex) {
    // Run LDA on the training data.
    pair<vector<vector<float>>, vector<int>> result = linearDiscriminantAnalysis(trainingData);
    vector<int> classOrder;
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

/******************************************************************
 * 10‑fold sweep over (k, threshold) pairs
 * ‑ kVals   : list of neighbour counts to test
 * ‑ tVals   : list of similarity‑threshold multipliers to test
 * Returns {bestK, bestT, bestAcc}
 ******************************************************************/
tuple<int,float,float> findBestParameters(vector<vector<vector<float>>> &dataset, vector<int> kVals = vector<int>{3, 5, 7, 9}, vector<float> tVals = vector<float>{0.15f, 0.20f, 0.25f, 0.30f}, int removalCount = 5, bool hidePrinting = false, int blockLevel = 1) {

    if (dataset.empty()) {
        cerr<<"Empty dataset\n";
        return make_tuple(-1,-1.f,-1.f);
    }

    /* -------- silence console if requested -------- */
    streambuf *oldBuf=nullptr; ostringstream sink;
    if (hidePrinting) oldBuf = cout.rdbuf(sink.rdbuf());

    const int FOLDS = 10;
    auto folds = DataUtil::splitDataset(dataset,FOLDS);

    /* accuracy accumulator indexed by [kIdx][tIdx] */
    vector<vector<float>> acc(kVals.size(),vector<float>(tVals.size(),0.f));

    for (int i=0;i<FOLDS;++i)
    {
        /* -------- build train / test split -------- */
        vector<vector<vector<float>>> train(NUM_CLASSES), test  = folds[i];

        for (int f=0;f<FOLDS;++f)
            if (f!=i)
                for (int c=0;c<NUM_CLASSES;++c)
                    train[c].insert(train[c].end(),folds[f][c].begin(),folds[f][c].end());

        /* -------- generate & simplify blocks -------- */
        vector<HyperBlock> hbs;
        vector<vector<float>> bestVecs;
        vector<vector<int>> bestIdx(NUM_CLASSES,vector<int>(FIELD_LENGTH));
        vector<int> eachBest(NUM_CLASSES);
        computeLDAOrdering(train,bestVecs,bestIdx,eachBest);

        IntervalHyperBlock::generateHBs(train,hbs,eachBest, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
        Simplifications::REMOVAL_COUNT = removalCount;

        // make a copy of our input data so that we don't break it for the KNN.
        vector<vector<vector<float>>> levelNTrain = train;

        // increase our block level until we hit the level we want.
        for (int level = 1; level < blockLevel; level++) {
            vector<HyperBlock> newBlocks;

            levelNTrain = move(IntervalHyperBlock::generateNextLevelHBs(levelNTrain, hbs, newBlocks, eachBest, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS));

            // knn actually does better when we are removing the extra points sometimes, so we use the original data and shrink the set does better KNN
            // train = move(IntervalHyperBlock::generateNextLevelHBs(train, hbs, newBlocks, eachBest, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS));

            hbs = move(newBlocks);
        }

        Simplifications::runSimplifications(hbs,train,bestIdx);

        /* -------- evaluate every (k,threshold) combo -------- */
        for (size_t kI=0;kI<kVals.size();++kI)
            for (size_t tI=0;tI<tVals.size();++tI) {
                Knn::deviationsComputed = false;            // reset per fold
                map<pair<int,int>,PointSummary> summaries;
                float foldAcc = testAccuracyOfHyperBlocks(hbs, test, train, summaries, kVals[kI], tVals[tI]);
                acc[kI][tI] += foldAcc;
            }
    }

    /* -------- compute averages & find best -------- */
    int    bestK  = -1;
    float  bestT  = -1.f;
    float  bestAcc= -1.f;

    for (size_t kI=0;kI<kVals.size();++kI)
        for (size_t tI=0;tI<tVals.size();++tI)
        {
            float avg = acc[kI][tI] / static_cast<float>(FOLDS);
            if (avg > bestAcc)
            { bestAcc = avg; bestK = kVals[kI]; bestT = tVals[tI]; }
            cout<<"K="<<kVals[kI]<<"  T="<<tVals[tI]
                     <<"  avgAcc="<<avg<<"\n";
        }

    if (hidePrinting) cout.rdbuf(oldBuf);

    cout<<"BEST -> K = "<<bestK<<"\tThreshold = "<< bestT <<"\taccuracy = " << bestAcc <<"\n";
    return make_tuple(bestK,bestT,bestAcc);
}

vector<float> runKFold(vector<vector<vector<float>>> &dataset, vector<pair<int,int>>& classPairs, bool oneToMany = true, bool takeUserInput = false, int removalCount = 5, int nearestNeighborK = 5, float similarityThreshold = 0.25f, bool hidePrinting = false) {

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
    if (hidePrinting) {
        ostringstream nullSink;
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

        // little thing. causes issues if we don't reset when we make a new training set
        Knn::deviationsComputed = false;

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
        vector<HyperBlock> hyperBlocks;

        if (oneToMany) {
            // ------------------------------------------
            // GENERATING BLOCKS BUSINESS AS USUAL
            IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);

            // simplify them, with the simplification count we have specifed as a parameter. usually 0, but playing with this value can get us better results because we are removing more blocks
            Simplifications::REMOVAL_COUNT = removalCount;
            //vector<int> result = Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);

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

            // get our accuracy now for this fold.
            map<pair<int, int>, PointSummary> pointSummaries;
            acc += testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData,pointSummaries, nearestNeighborK, similarityThreshold);
            blockCount += hyperBlocks.size();
            cCount += clauseCount;

            cout << "Block count: " << hyperBlocks.size() << endl;

        } // end of one train/test loop
        else {
            vector<vector<HyperBlock>> oneToOneBlocks = oneToOneHyper(trainingData, eachClassBestVectorIndex, classPairs);
            evaluateOneToOneHyperBlocks(oneToOneBlocks, testData, classPairs, NUM_CLASSES);
        }

    } // end of one train/test loop

    float avgAcc = float (acc) / float(k);
    float blockAvg = float(blockCount) / float(k);
    float clauseAvg = float(cCount) / float(k);

    if (hidePrinting)
        cout.rdbuf(oldBuf);           // back to console

    cout << "OVERALL ACCURACY " << avgAcc << endl;
    cout << "Average block count " << blockAvg << endl;
    cout << "Average clause count " << clauseAvg << endl;

    return {avgAcc, blockAvg, clauseAvg};
}

vector<float> runKFoldWithLevelNBlocks(vector<vector<vector<float>>> &dataset, bool takeUserInput = false, int removalCount = 0, int nearestNeighborK = 5, float similarityThreshold = 0.25f, bool hidePrinting = false, const int HB_LEVEL = 2) {

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
    if (hidePrinting) {
        ostringstream nullSink;
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

        // little thing. causes issues if we don't reset when we make a new training set
        Knn::deviationsComputed = false;

        // The test dataset for this iteration is simply fold i.
        vector<vector<vector<float>>> testData = kFolds[i];

        // now that our data is set up with training and testing, we simply do business as usual. we are going to do our LDA on the train data, then just do our block generation and simplification
        // Run LDA on the training data.
        vector<vector<float>>bestVectors;
        // Initialize indexes for each class.
        vector<vector<int>> bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
        vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);
        computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);

        cout << "----------------------------FOLD " << (i + 1) << " RESULTS----------------------------------" << endl;
        vector<HyperBlock> hyperBlocks;

        // ------------------------------------------
        // GENERATING BLOCKS BUSINESS AS USUAL
        IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);

        // now we iteratively increase the level of the blocks to whatever level
        vector<vector<vector<float>>> levelNData = trainingData;

        for (int level = 1; level < HB_LEVEL; level++) {
            vector<HyperBlock> thisLevelBlocks;
            // make our new set of blocks, and save this set of envelope cases. now we can reduce the training set iteratively.
            levelNData = move(IntervalHyperBlock::generateNextLevelHBs(levelNData, hyperBlocks, thisLevelBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS));

            // updating the train data itself actually allows us to perform better. we shrink the training set, and the KNN does better in this way.
            // trainingData = move(IntervalHyperBlock::generateNextLevelHBs(trainingData, hyperBlocks, thisLevelBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS));

            hyperBlocks  = move(thisLevelBlocks);   // advance to new level
        }

        // simplify them, with the simplification count we have specifed as a parameter. usually 0, but playing with this value can get us better results because we are removing more blocks
        Simplifications::REMOVAL_COUNT = removalCount;
        Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);

        int totalPoints = 0;
        for (const auto &c : trainingData)
            totalPoints += c.size();

        // clause count computed here because sometimes we don't simplify
        int clauseCount = 0;
        for (const auto &hb : hyperBlocks) {
            for (int a = 0; a < FIELD_LENGTH; a++) {
                if (hb.minimums[a][0] != 0.0f || hb.maximums[a][0] != 1.0f)
                    clauseCount++;
            }
        }

        // get our accuracy now for this fold.
        map<pair<int, int>, PointSummary> pointSummaries;
        acc += testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData,pointSummaries, nearestNeighborK, similarityThreshold);
        blockCount += hyperBlocks.size();
        cCount += clauseCount;
    } // end of one train/test loop

    float avgAcc = float (acc) / float(k);
    float blockAvg = float(blockCount) / float(k);
    float clauseAvg = float(cCount) / float(k);

    if (hidePrinting)
        cout.rdbuf(oldBuf);           // back to console if we had printing disabled.

    cout << "OVERALL ACCURACY " << avgAcc << endl;
    cout << "Average block count " << blockAvg << endl;
    cout << "Average clause count " << clauseAvg << endl;

    return {avgAcc, blockAvg, clauseAvg};

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
            cout << setw(5) << row[i] << " ";
        }
        cout << endl;
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

    // 3-D datasets
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
    const vector<vector<HyperBlock>>& oneToOneHBs,
    const vector<vector<vector<float>>>& testSet,
    const vector<pair<int, int>>& classPairs,
    int numClasses
) {
    int totalPoints = 0;
    int correctPoints = 0;

    vector<vector<int>> confusion(numClasses, vector<int>(numClasses, 0));
    vector<int> correctPerClass(numClasses, 0);
    vector<int> totalPerClass(numClasses, 0);


    for (int actualClass = 0; actualClass < numClasses; ++actualClass) {
        for (const auto& point : testSet[actualClass]) {
            vector<int> votes(numClasses, 0);

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
                        cerr << "Error: could not find block set for classes " << i << " and " << j << "\n";
                        continue;
                    }

                    const auto& pairHBs = oneToOneHBs[findPairIndex];


                    for (const auto& block : pairHBs) {
                        if (block.classNum == i && block.inside_HB(point.size(), point.data())) {
                            votes[i]++;
                        }
                        if (block.classNum == j && block.inside_HB(point.size(), point.data())) {
                            votes[j]++;
                        }
                    }
                }
            }

            // Majority vote
            int predictedClass = distance(votes.begin(), max_element(votes.begin(), votes.end()));
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
    cout << "\nConfusion Matrix:\n";
    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            cout << confusion[i][j] << "\t";
        }
        cout << "\n";
    }

    cout << "\nPer-Class Accuracy:\n";
    for (int i = 0; i < numClasses; ++i) {
        float acc = totalPerClass[i] ? static_cast<float>(correctPerClass[i]) / totalPerClass[i] : 0.0f;
        cout << "Class " << i << ": " << acc * 100.0f << "%\n";
    }

    cout << "\nOverall Accuracy: " << (static_cast<float>(correctPoints) / totalPoints) * 100.0f << "%\n";
}



/**
 * This function is a experimental one to test how weighting the votes of HBs by their performance
 * on a validation set. At a high level what we do in this is:
 *
 * 1. Split the training dataset into "reducedTraining" and "validationData"
 * 2. Generate HBs using the "reducedTraining" dataset.
 * 3. Evaluate the HBs using the "validationData" as a testing set. Within this we build PointSummaries
 *    which links every block index to the points from validation which fell inside the block. This allows
 *    for metrics like precision to be kept. Specifically, each block stores, its precision AND the amount of precision
 *    that each other class of points caused it to lose. The latter gives us a prediction of what the block might misclassify
 *    as belonging to its own class. Ex. high loss of precision from class y, means this blocks might actually vote for a y.
 *
 * 4. We save the afforementioned stats, then evaluate the HBs on the TRUE TESTING dataset using the precision weighted voting.
 * 5. Return the accuracy obtained.
 *
 *
 * @param trainingData  The training dataset input by the user.
 * @param eachClassBestVectorIndex Will be NUM_CLASSES long. Stores the best attribute to sort each class by during intervalHyper generation process.
 *                                 basically the most separating attribute for the specific class.
 * @param hyperBlocks              Empty Hyperblocks array, will be full when the function exits.
 * @param testingData      The dataset to be used during the final testing phase of the precision weighted hbs.
 * @param bestVectorsIndexes The order of attributes for each class. ex: {{0,1,2}, {2, 1, 0}} the attribute removal order
 *                           when simplifications are run for class 1 would be 0,1,2. The second class would be 2,1,0
 * @return  The accuracy, could be the stat struct if confusion matrix function is updated.
 */
float genAndRunPrecisionWeightedHBs(vector<vector<vector<float>>>& trainingData,
                                    vector<int> eachClassBestVectorIndex, vector<HyperBlock>& hyperBlocks,
                                    vector<vector<vector<float>>>& testingData,
                                    vector<vector<int>> bestVectorsIndexes
) {

    map<pair<int, int>, PointSummary> pointSummaries;

    //TODO: We need to set up a temp so that the training data is reset to its entire version after running this program.
    vector<vector<vector<float>>> validationData;
    DataUtil::createValidationSplit( trainingData, validationData, .10, 42);

    // Build the hbs
    IntervalHyperBlock::generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS);
    Simplifications::runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);

    // Test the validation HBS, returns confusion matrix, vector<vector<long>>
    vector<vector<vector<float>>> stillUnclassified(NUM_CLASSES);
    vector<vector<long>> confusionMatrix = ClassificationTests::buildConfusionMatrix(hyperBlocks, trainingData, validationData, ClassificationTests::HYPERBLOCKS,stillUnclassified , NUM_CLASSES, pointSummaries);

    for(auto& hb : hyperBlocks) {
        hb.setHBPrecisions(pointSummaries, NUM_CLASSES);
    }

    // Go through and make a non-distinct confusion matrix.
    std::vector<std::vector<long>> ultraConfusionMatrix(NUM_CLASSES, std::vector<long>(NUM_CLASSES, 0));

    for (const auto& entry : pointSummaries) {
        const PointSummary& summary = entry.second;
        int trueClass = summary.classIdx;

        for (const BlockInfo& hit : summary.blockHits) {
            int blockClass = hit.blockClass;
            ultraConfusionMatrix[trueClass][blockClass] += 1;
        }
    }


    pointSummaries.clear();
    vector<vector<long>> newConfusion = ClassificationTests::buildConfusionMatrix(hyperBlocks, trainingData, testingData, ClassificationTests::PRECISION_WEIGHTED, stillUnclassified , NUM_CLASSES, pointSummaries);
    cout << "\nPrecision Weighted Matrix " << endl;
    PrintingUtil::printConfusionMatrix(newConfusion, NUM_CLASSES, CLASS_MAP_INT);

    vector<vector<vector<float>>> unclassed(NUM_CLASSES);
    vector<vector<long>> knnMatrix = ClassificationTests::buildConfusionMatrix(hyperBlocks, trainingData, stillUnclassified, ClassificationTests::PURE_KNN, unclassed , NUM_CLASSES, pointSummaries);
    cout << "\nKNN matrix" << endl;
    PrintingUtil::printConfusionMatrix(knnMatrix, NUM_CLASSES, CLASS_MAP_INT);

    for(int i = 0; i < knnMatrix.size(); ++i) {
        for (int j = 0; j < knnMatrix[i].size(); ++j) {
            newConfusion[i][j] += knnMatrix[i][j];
        }
    }
    cout << "\nOld Matrix" << endl;
    vector<vector<long>> oldConf = ClassificationTests::buildConfusionMatrix(hyperBlocks, trainingData, testingData, ClassificationTests::HYPERBLOCKS, stillUnclassified , NUM_CLASSES, pointSummaries);
    PrintingUtil::printConfusionMatrix(oldConf, NUM_CLASSES, CLASS_MAP_INT);

    return PrintingUtil::printConfusionMatrix(newConfusion, NUM_CLASSES, CLASS_MAP_INT);
}



vector<float> precisionKFold(vector<vector<vector<float>>> &dataset, int nearestNeighborK = 5, float similarityThreshold = 0.25f, bool hidePrinting = false) {
    if (dataset.empty()) {
        cout << "Please enter a training dataset before using K Fold validation" << endl;
        return {-1, -1, -1};
    }

    int k = 10;
    streambuf* oldBuf = nullptr;
    if (hidePrinting) {
        ostringstream nullSink;
        oldBuf = cout.rdbuf(nullSink.rdbuf());
    }

    vector<vector<vector<vector<float>>>> kFolds = DataUtil::splitDataset(dataset, k);
    float acc = 0.0f;
    int blockCount = 0;
    int cCount = 0;

    for (int i = 0; i < k; i++) {
        vector<vector<vector<float>>> trainingData(NUM_CLASSES);

        for (int fold = 0; fold < k; fold++) {
            if (fold == i) continue;
            for (int cls = 0; cls < NUM_CLASSES; cls++) {
                trainingData[cls].insert(trainingData[cls].end(), kFolds[fold][cls].begin(), kFolds[fold][cls].end());
            }
        }

        vector<vector<vector<float>>> testData = kFolds[i];

        vector<vector<float>> bestVectors;
        vector<vector<int>> bestVectorsIndexes(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
        vector<int> eachClassBestVectorIndex(NUM_CLASSES);
        computeLDAOrdering(trainingData, bestVectors, bestVectorsIndexes, eachClassBestVectorIndex);

        cout << "----------------------------FOLD " << (i + 1) << " RESULTS----------------------------------" << endl;

        vector<HyperBlock> hyperBlocks;

        acc += genAndRunPrecisionWeightedHBs(trainingData, eachClassBestVectorIndex, hyperBlocks, testData, bestVectorsIndexes);

        int clauseCount = 0;
        for (const auto &hb : hyperBlocks) {
            for (int a = 0; a < FIELD_LENGTH; a++) {
                if (hb.minimums[a][0] != 0.0f || hb.maximums[a][0] != 1.0f)
                    clauseCount++;
            }
        }

        blockCount += hyperBlocks.size();
        cCount += clauseCount;

        cout << "Block count: " << hyperBlocks.size() << endl;
    }

    float avgAcc = acc / k;
    float blockAvg = static_cast<float>(blockCount) / k;
    float clauseAvg = static_cast<float>(cCount) / k;

    if (hidePrinting)
        cout.rdbuf(oldBuf);

    cout << "OVERALL ACCURACY " << avgAcc << endl;
    cout << "Average block count " << blockAvg << endl;
    cout << "Average clause count " << clauseAvg << endl;

    return {avgAcc, blockAvg, clauseAvg};
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
    vector<pair<int, int>> classPairsOut;

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
                    system("dir datasets");
                #else
                    system("ls datasets");
                #endif
                getline(cin, trainingDataFileName);
                string fullPath = "datasets" + string(PATH_SEPARATOR) + trainingDataFileName;
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
                    minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
                    maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());
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

                // this has to get set false each time we make a new dataset or else we are going to make a seg fault when we test a second dataset.
                Knn::deviationsComputed = false;

                PrintingUtil::waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                cout << "Enter testing data filename: " << endl;
                system("ls datasets");
                getline(cin, testingDataFileName);
                string fullPath = "datasets" + string(PATH_SEPARATOR) + testingDataFileName;

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

                for(HyperBlock& hb: hyperBlocks){
                  hb.find_avg_and_size(trainingData);
                }

                PrintingUtil::waitForEnter();
                break;

            }
            case 5: { // EXPORT HYPERBLOCKS
                cout << "Enter the file to save HyperBlocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                DataUtil::saveBasicHBsToBinary(hyperBlocks, hyperBlocksExportFileName, FIELD_LENGTH);
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
            case 7: {       // SIMPLIFY HYPERBLOCKS
             
                // testing for time and to determine if they are doing the same.
                auto start = chrono::high_resolution_clock::now();
                Simplifications::removeUselessAttrNoDisjunction(hyperBlocks, trainingData, bestVectorsIndexes);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> diff = end - start;
                cout << "Time taken: " << diff.count() << " seconds" << endl;

                int newClauseCount = 0;
                for (const auto &hb : hyperBlocks) {
                    for (int a = 0; a < FIELD_LENGTH; a++) {
                        if (hb.minimums[a][0] != 0.0f || hb.maximums[a][0] != 1.0f)
                            newClauseCount++;
                    }
                }

                cout << "New clause count: " << newClauseCount << endl;

                int totalPoints = 0;
                for (const auto &c : trainingData) totalPoints += c.size();

                cout << "After removing useless blocks we have: " << newClauseCount << " clauses\n";
                cout << "We got a final total of: " << hyperBlocks.size() << " blocks." << endl;
                cout << "We had: " << totalPoints << " points of training data\n";
                PrintingUtil::waitForEnter();
                break;
            }
            case 8: { // TEST HBs ON CURRENT TESTING DATASET
                cout << "Testing HBs on testing dataset" << endl;
                map<pair<int, int>, PointSummary> pointSummaries;
                testAccuracyOfHyperBlocks(hyperBlocks, testData, trainingData, pointSummaries);
                PrintingUtil::waitForEnter();
                break;
            }
            case 9: {   // TEST 1-1 HBs ON CURRENT TESTING DATASET
                evaluateOneToOneHyperBlocks(oneToOneBlocks, testData, classPairsOut, NUM_CLASSES);
                PrintingUtil::waitForEnter();
                break;
            }
            case 10: {  // RUN K-FOLD CROSS VALIDATION
                vector<pair<int,int>> classPairs{}; // just needed because we have to pass in something

                // run the k fold, taking the user input for number of k. using default values for removal count, k and whatnot
                runKFold(trainingData, classPairs, true, true);  //precisionKFold(trainingData);
                PrintingUtil::waitForEnter();
                break;
            }
            case 11: {  // GENERATE 1-1 HBs
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
            case 12: {    // IMPORT 1-1 HBs
                cout << "Enter 1-1 Hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);
                oneToOneBlocks = DataUtil::loadOneToOneHBsFromBinary(hyperBlocksImportFileName, classPairsOut);
                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;

                PrintingUtil::waitForEnter();
                break;
            }
            case 13: { // EXPORT 1-1 HBs
                cout << "Enter the file to save Hyperblocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                DataUtil::saveOneToOneHBsToBinary(oneToOneBlocks, hyperBlocksExportFileName);
                break;
            }
            case 14: {  // RUN K-FOLD USING THE 1-1 HBs
                vector<pair<int,int>> classPairs{}; // need to make up the class pairings

                runKFold(trainingData, classPairs, false, true);
                PrintingUtil::waitForEnter();
                break;
            }
            case 15: {  // GENERATE 1-Rest HBs
                // POINT BASED ORDER
                oneToRestBlocks.clear();


                auto start = chrono::high_resolution_clock::now();
                oneToRestBlocks = oneToRestHyper(trainingData, eachClassBestVectorIndex);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> diff = end - start;

                // Flatten the set of HBs by moving them.
                vector<HyperBlock> allRestBlocks;
                for (auto& blockSet : oneToRestBlocks) {
                    allRestBlocks.insert(allRestBlocks.end(),make_move_iterator(blockSet.begin()),make_move_iterator(blockSet.end()));
                }

                DataUtil::saveBasicHBsToBinary(allRestBlocks, "digitBlocksRest.csv", FIELD_LENGTH);
                cout << "Finished Generating one to Some blocks." << endl;
                cout << "Elapsed time: " << diff.count() << " seconds\n";
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

                cout << "What level HBs are we testing?" << endl;
                int blockLevel;
                cin >> blockLevel;
                cin.ignore(numeric_limits<streamsize>::max(), '\n');

                if (cin.fail() || maxK < 0) {
                    cout << "Error: Invalid input. Please enter a valid integer greater than 1." << endl;
                    // Clear the error state and ignore any remaining input.
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    return;
                }

                vector<int> removalCounts;
                for (int i = 0; i <= maxRemoval; i++) {
                    removalCounts.push_back(i);
                }

                vector<int> kVals;
                for (int i = 1; i <= maxK; i += 2) {
                    kVals.push_back(i);
                }

                vector<float> thresholds{0.15, 0.2, 0.25, 0.3};

                // findBestParameters(trainingData, maxRemoval, maxK);
                findBestParameters(trainingData, kVals, thresholds, maxRemoval, false, blockLevel);
                PrintingUtil::waitForEnter();
                break;
            }
            case 17: {
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    PrintingUtil::waitForEnter();
                } else {
                    vector<HyperBlock> newBlocks;
                    trainingData = move(IntervalHyperBlock::generateNextLevelHBs(trainingData, hyperBlocks, newBlocks, eachClassBestVectorIndex, FIELD_LENGTH, COMMAND_LINE_ARGS_CLASS));
                    hyperBlocks = move(newBlocks);
                }
                static int levelN = 1;
                cout << "Finished Generating level " << ++levelN << " HBs" << endl;
                PrintingUtil::waitForEnter();
                break;
            }
            case 18: {
                // run our level N k fold function
                runKFoldWithLevelNBlocks(trainingData, false, 1, 1, .25f);
                PrintingUtil::waitForEnter();
                break;
            }
            case 19: {
                // MERGE FIX: This was going to be in 17, but level N hbs displaces it
                genAndRunPrecisionWeightedHBs(trainingData, eachClassBestVectorIndex, hyperBlocks , testData, bestVectorsIndexes);
                PrintingUtil::waitForEnter();
                break;
            }
            case 20: {
                running = false;
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