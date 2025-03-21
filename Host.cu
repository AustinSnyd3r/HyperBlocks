#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <future>
#include "CudaUtil.h"
#include "HyperBlock.h"
#include "HyperBlockCuda.cuh"
#include "LDA.cpp"
#include <chrono>
#include <omp.h>
#include <iomanip>  // For setw
#include <algorithm> // For max
#include <queue>
using namespace std;

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset
int COMMAND_LINE_ARGS_CLASS = -1;

map<string, int> CLASS_MAP;
map<string, int> CLASS_MAP_TESTING;

map<int, string> CLASS_MAP_INT;
map<int, string> CLASS_MAP_TESTING_INT;

// Struct version of DataATTR xrecord
struct DataATTR {
    float value; // Value of one attribute of a point
    int classNum; // The class number of the point
    int classIndex; // The index of point within the class

    DataATTR(float val, int cls, int index) : value(val), classNum(cls), classIndex(index) {}
};

// Interval struct to make interval thing more understandable
struct Interval{
    int size;
    int start;
    int end;
    int attribute;

    Interval(int s, int st, int e, int a) : size(s), start(st), end(e), attribute(a) {}
};

Interval longestInterval(vector<DataATTR>& dataByAttribute, float accThreshold, vector<HyperBlock>& existingHB, int attr);
void removeValueFromInterval(vector<DataATTR>& dataByAttribute, Interval& intr, float value);
int skipValueInInterval(vector<DataATTR>& dataByAttribute, int i, float value);
bool checkIntervalOverlap(vector<DataATTR>& dataByAttribute, Interval& intr, int attr, vector<HyperBlock>& existingHB);
void merger_cuda(const vector<vector<vector<float>>>& dataWithSkips, const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks);
void saveBasicHBsToCSV(const vector<HyperBlock>& hyperBlocks, const string& fileName);
void printDataset(const vector<vector<vector<float>>>& vec);
void printConfusionMatrix(vector<vector<long>>& data);
float euclideanDistance(const vector<float>& vec1, const vector<float>& vec2);
vector<vector<long>> kNN(vector<vector<vector<float>>> unclassifiedData, vector<HyperBlock>& hyperBlocks, int k);

// Function to reorder testing dataset based on training class mapping
vector<vector<vector<float>>> reorderTestingDataset(
    const vector<vector<vector<float>>>& testingData,
    const map<string, int>& CLASS_MAP_TRAINING,
    const map<string, int>& CLASS_MAP_TESTING
) {
    // Create a new vector with the same size as the testing data
    vector<vector<vector<float>>> reorderedTestingData(testingData.size());

    // Create a mapping from testing indices to training indices
    map<int, int> indexMap;
    // For each (className → trainingIndex) in the training map
    for (const auto& entry : CLASS_MAP_TRAINING) {
        const std::string& className   = entry.first;
        int trainingIndex = entry.second;

        // Find the same class in the testing map
        auto it = CLASS_MAP_TESTING.find(className);
        if (it != CLASS_MAP_TESTING.end()) {
            int testingIndex = it->second;
            indexMap[testingIndex] = trainingIndex;

            // Ensure our output vector is big enough
            if (trainingIndex >= reorderedTestingData.size()) {
                reorderedTestingData.resize(trainingIndex + 1);
            }
        }
    }


    // Reorder the testing data
    for (int testingIndex = 0; testingIndex < testingData.size(); testingIndex++) {
        auto it = indexMap.find(testingIndex);
        if (it != indexMap.end()) {
            int trainingIndex = it->second;
            reorderedTestingData[trainingIndex] = testingData[testingIndex];
        }
    }

    return reorderedTestingData;
}

/***
* We want to go through the hyperBlocks that were generated and write them to a file.
*
*
* This print isn't caring about disjunctive blocks.
*/
void saveBasicHBsToCSV(const vector<HyperBlock>& hyperBlocks, const string& fileName){
	// Open file for writing
    ofstream file(fileName);
    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return;
    }

	// min1, min2, min3, ..., minN, max1, max2, max3, ..., maxN, class
	for (const auto& hyperBlock : hyperBlocks) {
        // Write minimums
        for (const vector<float>& min : hyperBlock.minimums) {
            file << min[0] << ",";
        }

        // Write maximums
        for (const vector<float>& max : hyperBlock.maximums) {
            file << max[0] << ",";
        }

        // Write the class number
        file << hyperBlock.classNum << "\n";
    }

    file.close();
}

///////////////////////// FUNCTIONS FOR intervalHyper IMPLEMENTATION /////////////////////////

 /**
     * Finds largest interval across all dimensions of a set of data.
     * @param dataByAttribute all data split by attribute
     * @param accThreshold accuracy threshold for interval
     * @param existingHB existing hyperblocks to check for overlap
     * @return largest interval
     */
vector<DataATTR> intervalHyper(vector<vector<DataATTR>>& dataByAttribute, float accThreshold, vector<HyperBlock>& existingHB){
    //cout << "Starting interval hyperblock" << endl;
    vector<future<Interval>> intervals;
    int attr = -1;
    Interval best(-1, -1, -1, -1);

   // Search each attribute
    for (int i = 0; i < dataByAttribute.size(); i++) {
        // Launch async task
        intervals.emplace_back(async(launch::async, longestInterval, ref(dataByAttribute[i]), accThreshold, ref(existingHB), i));
    }

    // Wait for results then find largest interval
    for(auto& future1 : intervals){
        Interval intr = future1.get();
        if(intr.size > 1 && intr.size > best.size){
            best.size = intr.size;
            best.start = intr.start;
            best.end = intr.end;
            best.attribute = intr.attribute;

            attr = intr.attribute;
        }
    }
	//cout << "Best.start: " << best.start << "  Best.end: " << best.end <<"  Best.size: " << best.size <<  "  Best.attribute: " << best.attribute << endl;

    // Construct ArrayList of data
    vector<DataATTR> longest;
    if(best.size != -1){
        for(int i = best.start; i <= best.end; i++){
          	//cout << "Data by attribute printing time: " << dataByAttribute[attr][i].classNum << " " << dataByAttribute[attr][i].classIndex << "\n";
            //cout << attr << endl;

            longest.push_back(dataByAttribute[attr][i]);
        }
    }
    //cout << "Finished interval hyperblock" << endl;

    return longest;
}

/**
 * Seperates data into seperate vecs by attribute
 */
vector<vector<DataATTR>> separateByAttribute(vector<vector<vector<float>>>& data){
    vector<vector<DataATTR>> attributes;

    // Go through the attribute columns
    for(int k = 0; k < FIELD_LENGTH; k++){
        vector<DataATTR> tmpField;

        // Go through the classes
        for(int i = 0; i < data.size(); i++){
            // Go through the points
            for(int j = 0; j < data[i].size(); j++){
                tmpField.push_back(DataATTR(data[i][j][k], i, j));
            }
        }
    
        // Sort data by value then add
        sort(tmpField.begin(), tmpField.end(), [](const DataATTR& a, const DataATTR& b) {
            return a.value < b.value;
        });
        attributes.push_back(tmpField);
    }

    return attributes;
}

/***
 * This will sort the array based on the "best" columns values
 * 
 * The columns themselves aren't moving, we are moving the points
 * based on the one columns values;
 */
void sortByColumn(vector<vector<float>>& classData, int colIndex) {
    sort(classData.begin(), classData.end(), [colIndex](const vector<float>& a, const vector<float>& b) {
        return a[colIndex] < b[colIndex];
    });
}

/***
 * Finds the longest interval in a sorted list of data by attribute.
 * @param dataByAttribute sorted data by attribute
 * @param accThreshold accuracy threshold for interval
 * @param existingHB existing hyperblocks to check for overlap
 * @param attr attribute to find interval on
 * @return longest interval
*/
Interval longestInterval(vector<DataATTR>& dataByAttribute, float accThreshold, vector<HyperBlock>& existingHB, int attr){
    //cout << "Started longest interval \n" << endl;

    Interval intr(1, 0, 0, attr);
    Interval max_intr(-1, -1, -1, attr);

    int n = dataByAttribute.size();
    float misclassified = 0;

    for(int i = 1; i < n; i++){
        // If current class matches with next
        if(dataByAttribute[intr.start].classNum == dataByAttribute[i].classNum){
            intr.size++;
        }
        else if( (misclassified+1) / intr.size > accThreshold){
            // ^ i think this is a poor way to check. but not changing rn for the translation from java
            misclassified++;
            intr.size++;
        }
        else{
            // Remove value from interval if accuracy is below threshold.
            if(dataByAttribute[i-1].value == dataByAttribute[i].value){
                // remove then skip overlapped values
                removeValueFromInterval(dataByAttribute, intr, dataByAttribute[i].value);
                i = skipValueInInterval(dataByAttribute, i, dataByAttribute[i].value);
            }

            // Update longest interval if it doesn't overlap
            if(intr.size > max_intr.size && checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
                max_intr.start = intr.start;
                max_intr.end = intr.end;
                max_intr.size = intr.size;
                max_intr.attribute = attr;
            }

            // Reset curr interval
            intr.size = 1;
            intr.start = i;
            misclassified = 0;
        }
        intr.end = i;
    }

    // final check update longest interval if it doesn't overlap
    if(intr.size > max_intr.size && checkIntervalOverlap(dataByAttribute, intr, attr, existingHB)){
        max_intr.start = intr.start;
        max_intr.end = intr.end;
        max_intr.size = intr.size;
    }

    //cout << "Finished longest interval \n" << endl;

    return max_intr;
}


bool checkIntervalOverlap(vector<DataATTR>& dataByAttribute, Interval& intr, int attr, vector<HyperBlock>& existingHB){
    //cout << "Started check interval overlap\n" << endl;
    // interval range of vals
    float intv_min = dataByAttribute[intr.start].value;
    float intv_max = dataByAttribute[intr.end].value;
   
    /*
    *   check if interval range overlaps with any existing hyperblocks
    * to not overlap the interval maximum must be below all existing hyperblock minimums
    * or the interval minimum must be above all existing hyperblock maximums
    */
    for(const HyperBlock& hb : existingHB){
        if (!(intv_max < hb.minimums.at(attr).at(0) || intv_min > hb.maximums.at(attr).at(0))){
            return false;
        }
    }

    //cout << "Finished check interval overlap\n" << endl;

    // If unique return true
    return true;
}

//skipValueInInterval
int skipValueInInterval(vector<DataATTR>& dataByAttribute, int i, float value){
    //cout << "Starting skip value in interval\n" << endl;

    while(dataByAttribute[i].value == value){
        if(i < dataByAttribute.size() - 1){
            i++;
        }
        else{
            break;
        }
    }

    //cout << "Finished skip value in interval\n" << endl;

    return i;
}


//removeValueFromInterval
void removeValueFromInterval(vector<DataATTR>& dataByAttribute, Interval& intr, float value){
    //cout << "Starting remove value from intervals\n" << endl;
    while(dataByAttribute[intr.end].value == value){
        if(intr.end > intr.start){
            intr.size--;
            intr.end--;
        }
        else{
            intr.size = -1;
            break;
        }
    }
    //cout << "Finished remove value from intervals\n" << endl;
}

///////////////////////// END FUNCTIONS FOR intervalHyper IMPLEMENTATION /////////////////////////
void generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyperBlocks, vector<int> &bestAttributes){
  	// "Started generating HBS\n" << endl;
    // Hyperblocks generated with this algorithm
    vector<HyperBlock> gen_hb;

    // Get data to create hyperblocks
    vector<vector<DataATTR>> dataByAttribute = separateByAttribute(data);
    vector<vector<DataATTR>> all_intv;

    // Create dataset without data from interval HyperBlocks
    vector<vector<vector<float>>> datum;
    vector<vector<vector<float>>> seed_data;
    vector<vector<int>> skips;
	// "Initialized datum, seed_data, skips\n" << endl;

    // Initially generate blocks

    //cout << "Starting while loop to generate hyperblocks\n";
		//cout << "dataByAttribute[0].size() = " << dataByAttribute[0].size() << endl;

        while(dataByAttribute[0].size() > 0){
			//cout << "Attempting to go into intervalHyper " << endl;

            vector<DataATTR> intv = intervalHyper(dataByAttribute, 100, gen_hb);
        all_intv.push_back(intv);
        //cout << "Pushed to back of all intervals" << endl;

        // if hyperblock is unique then add
        if(intv.size() > 1){
            //cout << "making hb and intv_data" << endl;
            vector<vector<vector<float>>> hb_data;
            vector<vector<float>> intv_data;


            // Add the points from real data that are in the intervals
            for(DataATTR& dataAttr : intv){
                /*cout << "Trying to add a dataATTR. " << endl;

                if(dataAttr.classNum > 1 || dataAttr.classNum < 0){
                    cout << "Invalid classNum: " << dataAttr.classNum << endl;
                    cout << "Value: " << dataAttr.value << endl;
                    continue;
                }


                if(dataAttr.classIndex > data[dataAttr.classNum].size() - 1 || dataAttr.classIndex < 0){
                    cout << "Invalid class index: " << dataAttr.classIndex << endl;
                    continue;
                }
                */
                intv_data.push_back(data[dataAttr.classNum][dataAttr.classIndex]);
            }

            //cout << "Made it past the points from real data thingy" << endl << endl;
            // add data and hyperblock
            hb_data.push_back(intv_data);
            //cout << "Added intv data to hb_data" << endl << endl;

            HyperBlock hb(hb_data, intv[0].classNum);
            //cout << "Made the hyperblock for this interval thing" << endl << endl;

            gen_hb.push_back(hb);
                //cout << "Added results from last intervalHyper" << endl << endl;
        }else{
            //cout << "Breaking because the intv size is < 1" << endl;
            break;
        }
    }

        // Add all hbs from gen_hb to hyperBlocks
        hyperBlocks.insert(hyperBlocks.end(), gen_hb.begin(), gen_hb.end());

    // All data: go through each class and add points from data
    for(const vector<vector<float>>& classData : data){
        datum.push_back(classData);
        seed_data.push_back(vector<vector<float>>());
        skips.push_back(vector<int>());
    }

    // find which data to skip
    for(const vector<DataATTR>& dataAttrs : all_intv){
        for(const DataATTR& dataAttr : dataAttrs){
            skips[dataAttr.classNum].push_back(dataAttr.classIndex);
        }
    }
    // Sort the skips
    for(vector<int>& skip : skips){
        sort(skip.begin(), skip.end());
    }

    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(skips[i].size() > 0){
                if(j != skips[i][0]){
                    seed_data[i].push_back(data[i][j]);
                }
                else{
                    // remove first element from skips[i]
                    skips[i].erase(skips[i].begin());
                }
            }
            else{
                seed_data[i].push_back(data[i][j]);
            }
        }
    }

    // Sort data by most important attribute
    for(int i = 0; i < datum.size(); i++){
        sortByColumn(datum[i], bestAttributes[i]);
        sortByColumn(seed_data[i], bestAttributes[i]);
    }

    // Call CUDA function.
    //cout << "Calling merger_cuda\n" << endl;

    try{
        //cout << "Printing interval hyperblocks:\n\n" << endl;
        //for(const auto& hb : hyperBlocks){
        //    for(const auto& min : hb.minimums){
        //        cout << min[0] << " ";
        //    }
        //   cout << endl;
		//
        //    for(const auto& max : hb.maximums){
        //        cout << max[0] << " ";
        //    }
        //    cout << endl;
        //}
        //cout << "End interval hyperblocks:\n\n" << endl;
		//cout << "DATUM BEING PASSED INTO MERGING:" << endl;
		//printDataset(datum);

        //cout << "SEED DATA BEING PASSED INTO MERGING:" << endl;
		//printDataset(seed_data);

        merger_cuda(seed_data, datum, hyperBlocks);
    }catch (exception e){
        cout << "Error in generateHBs: merger_cuda" << endl;
    }
}

/*

    What I need to do.

    ********************************************************************************
     Find how many points and which points are classified into multiple classes
    ********************************************************************************


          - Right now I am using a set for each class and removing a point when it is classified, however
          - maybe i should do something else for this.

          - Maybe i want some form of set that maintains the use of each point index. However, we will also
          - keep track of where it was classified each time. So we could have the <pointClassIndex : Int, Pair(array[num_classes], totalClassifications: Int)>



    ************************************************
     Use the K-NN approach on unclassified points
    ************************************************

        Need a algorithm that will turn each HB into a single point, use the middle maybe.     (later do some weighting possibly? based on where the average actually is)

        We will essentially take unclassified points, and find the k nearest neighbor blocks, we will then check if there is a majority of any class in this.
        If there is we wil classify that point into the class that was a majority.

        (Can either refuse to classify points that don't have a majority or continue with a higher k value)

 */

void printDataset(const vector<vector<vector<float>>>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << "Class " << i << ":" << endl;
        for (const auto& row : vec[i]) {
            cout << "  [";
            for (int j = 0; j < row.size(); j++) {
                cout << row[j];
                if (j < row.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        cout << endl;  // Add spacing between classes
    }
}

/*  Returns a class seperated version of the dataset
 *  Each class has an entry in the outer vector with a 2-d vector of its points
 */
vector<vector<vector<float>>> dataSetup(const string filepath, map<string, int>& classMap, map<int, string>& reversedClassMap) {
    // 3D vector: data[class][point][attribute]
    vector<vector<vector<float>>> data;

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Failed to open file " << filepath << endl;
        return data;
    }

    int classNum = 0;
    string line;
    // Ignore the header, can use later if needed
    getline(file, line);

    // Read through all rows of CSV
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<string> row;

        // Read the entire row, splitting by commas
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Skip empty lines
        if (row.empty()) continue;

        string classLabel = row.back();
        row.pop_back();

        // Check if class exists, else create new entry
        if (classMap.count(classLabel) == 0) {
            classMap[classLabel] = classNum;
            data.push_back(vector<vector<float>>());
            classNum++;
        }

        int classIndex = classMap[classLabel];

        vector<float> point;
        for (const string& val : row) {
            try {
                point.push_back(stof(val));  // Convert to float and add to the point
            } catch (const invalid_argument&) {
                cerr << "Invalid value '" << val << "' in CSV" << endl;
                point.push_back(0.0f);  // Default to 0 if conversion fails
            }
        }

        // Add the points
        data[classIndex].push_back(point);
    }

    for (const auto& pair : classMap) {
        reversedClassMap[pair.second] = pair.first;
    }
    file.close();

    // Set global variables
    FIELD_LENGTH = data.empty() ? 0 : static_cast<int>(data[0][0].size());
    NUM_CLASSES = classNum;

    return data;
}


/* This needs to be a function to serialize hyperblocks.
 * take in 3-D vector that is the hyperblocks for each class
 * each class gets a dimension, with a 2-d vector for the HBs
 * assumes each row in the 2-D vector is 1 hyperblock
 * the first 1/2 of the row is the max's, the end is the mins.
 */
void saveHyperBlocksToFile(const string& filepath, const vector<vector<vector<float>>>& hyperBlocks) {
    ofstream file(filepath);

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filepath << endl;
        return;
    }

    // Loop through each class (outermost vector)
    for (int classNum = 0; classNum < hyperBlocks.size(); classNum++) {
        // Loop through each hyperblock (2D vector)
        for (const auto& hyperblock : hyperBlocks[classNum]) {
            // Write hyperblock values
            for (int i = 0; i < hyperblock.size(); i++) {
                file << hyperblock[i];
                if (i < hyperblock.size()) file << ", ";
            }
            // Append the class index
            file << ", " << classNum << "\n";
        }
    }

    file.close();
    cout << "Hyperblocks saved to " << filepath << endl;
}


vector<HyperBlock> loadBasicHBsFromCSV(const string& fileName) {
    ifstream file(fileName);
    vector<HyperBlock> hyperBlocks;

    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return hyperBlocks;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<vector<float>> minimums, maximums;
        string value;
        vector<float> temp_vals;

        while (getline(ss, value, ',')) {
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (!value.empty()) {
                temp_vals.push_back(stof(value));
            }
        }

        if (temp_vals.empty()) continue;

        int num_attributes = temp_vals.size() / 2;
        int classNum = static_cast<int>(temp_vals.back());
        temp_vals.pop_back(); // Remove classNum from the list

        for (int i = 0; i < num_attributes; ++i) {
            minimums.push_back({ temp_vals[i] });
            maximums.push_back({ temp_vals[i + num_attributes] });
        }

        hyperBlocks.emplace_back(maximums, minimums, classNum);
    }

    file.close();
    return hyperBlocks;
}

/**
* Find the min/max values in each column of data across the dataset.
* Can use this in normalization and also for making sure test set is normalized with
* the same values as the training set.
*/
void findMinMaxValuesInDataset(const vector<vector<vector<float>>>& dataset, vector<float>& minValues, vector<float>& maxValues) {
    // Step 1: Find min and max for each attribute
    for (const auto& class_data : dataset) {
        for (const auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                minValues[k] = min(minValues[k], point[k]);
                maxValues[k] = max(maxValues[k], point[k]);
            }
        }
    }
}

/**
* A function to normalize the test set using the given mins/maxes that were used to normalize the initial set
*/
void normalizeTestSet(vector<vector<vector<float>>>& testSet, const vector<float>& minValues, const vector<float>& maxValues) {
    if (testSet.empty()){
      cout << "Test set was empty when trying to normalize" << endl;
      return;
	}

    // Print out the min and max values first 20
    for (int i = 0; i < FIELD_LENGTH; i++) {
      cout << minValues[i] << endl;
    }
    cout << endl;

    cout << "Maxes" << endl;


    for (int i = 0; i < FIELD_LENGTH; i++) {
      cout << maxValues[i] << ",";
    }
    cout << endl;

    for (auto& class_data : testSet) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                if (maxValues[k] != minValues[k]) {
                    point[k] = (point[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                } else {
                    point[k] = 0.5f;
                }
            }
        }
    }
}

void minMaxNormalization(vector<vector<vector<float>>>& dataset, const vector<float>& minValues, const vector<float>& maxValues) {
    //cout << "Starting min-max normalization\n" << endl;

    if (dataset.empty()) return;

    int num_classes = dataset.size();

    // Min and max values for each attribute
    //vector<float> minValues(FIELD_LENGTH, numeric_limits<float>::infinity());
    //vector<float> maxValues(FIELD_LENGTH, -numeric_limits<float>::infinity());

    // Step 1: Find min and max for each attribute
	//findMinMaxValuesInDataset(dataset, minValues, maxValues);

    // Step 2: Apply Min-Max normalization
    for (auto& class_data : dataset) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                // Avoid div/0
                if (maxValues[k] != minValues[k]) {
                    point[k] = (point[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                } else {
                    //cout << "Column found with useless values" << endl;
                    point[k] = 0.5f;
                }
            }
        }
    }
    //cout << "Finished min-max normalization\n" << endl;
}

vector<bool> markUniformColumns(const vector<vector<vector<float>>>& data) {
     // cout << "Starting mark uniform columns\n" << endl;

    if (data.empty() || data[0].empty()) return vector<bool>(); // Handle edge case

    int numCols = data[0][0].size();
    vector<bool> removed(numCols, false);

    // Iterate through each column
    for (int col = 0; col < numCols; col++) {
        float referenceValue = data[0][0][col]; // Use first row of first class as reference
        bool allSame = true;

        // Check across all classes and all rows
        for (const auto& obj : data) {
            for (const auto& row : obj) {
                if (row[col] != referenceValue) {
                    allSame = false;
                    break;
                }
            }
            if (!allSame) break;
        }

        // If the column is uniform across all classes, mark it for removal
        if (allSame) {
            removed[col] = true;
        }
    }

   // cout << "Finished mark uniform columns\n" << endl;

    return removed;
}

// Source
void merger_cuda(const vector<vector<vector<float>>>& dataWithSkips, const vector<vector<vector<float>>>& allData, vector<HyperBlock>& hyperBlocks) {

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : allData) {
        numPoints += classData.size();
    }

    // Count blocks per class
    vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (const auto& hb : hyperBlocks) {
        numBlocksOfEachClass[hb.classNum]++;
    }

    vector<vector<HyperBlock>> resultingBlocks(NUM_CLASSES);
    
    // get our device count, and max it so that the most we will use is numClasses if we have only 2 classes for example.
    /* MULTI GPU BUSINESS
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    deviceCount = min(deviceCount, NUM_CLASSES);


    // Process each class
    // we have our multithreading happen here at this level. set the device to class % deviceCount
    #pragma omp parallel for num_threads(deviceCount)
    */

    int PADDED_LENGTH = ((FIELD_LENGTH + 3) / 4) * 4;
    // Find best occupancy
    int sharedMemSize = 2 * PADDED_LENGTH * sizeof(float);
    int minGridSize, blockSize;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mergerHyperBlocks, sharedMemSize, 0);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaOccupancyMaxPotentialBlockSize: %s\n", cudaGetErrorString(err));
        exit(-1);
    }


    int temp = 0;
    int goToClass = NUM_CLASSES;
    if (COMMAND_LINE_ARGS_CLASS != -1){
         temp = COMMAND_LINE_ARGS_CLASS;
         goToClass = COMMAND_LINE_ARGS_CLASS + 1;
    }

    for (int classN = temp; classN < goToClass; classN++) {
        
        // set our device based on class. this way even single threaded we use multiple GPUs
        // MORE MULTI GPU BUSINESS
        //cudaSetDevice(classN % deviceCount);

        int totalDataSetSizeFlat = numPoints * PADDED_LENGTH;
        int sizeWithoutHBpoints = ((dataWithSkips[classN].size() + numBlocksOfEachClass[classN]) * PADDED_LENGTH);
        if (dataWithSkips[classN].empty()) {
            sizeWithoutHBpoints = numBlocksOfEachClass[classN] * PADDED_LENGTH;
        }

        // Compute grid size to cover all elements. we already know our ideal block size from before.
        int gridSize = ((sizeWithoutHBpoints / PADDED_LENGTH) + blockSize - 1) / blockSize;

        #ifdef DEBUG
        cout << "Grid size: " << gridSize << endl;
        cout << "Block size: " << blockSize << endl;
        cout << "Shared memory size: " << sharedMemSize << endl;
        #endif

        // Allocate host memory
        vector<float> hyperBlockMinsC(sizeWithoutHBpoints);
        vector<float> hyperBlockMaxesC(sizeWithoutHBpoints);
        vector<int> deleteFlagsC(sizeWithoutHBpoints / PADDED_LENGTH);

        int nSize = allData[classN].size();
        vector<float> pointsC(totalDataSetSizeFlat - (nSize * PADDED_LENGTH));

        // Fill hyperblock arrays
        int currentClassIndex = 0;
        for (int currentClass = 0; currentClass < dataWithSkips.size(); currentClass++) {
            for (const auto& point : dataWithSkips[currentClass]) {
                if (currentClass == classN) {
                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        //if (removed[attr]) continue;
                        hyperBlockMinsC[currentClassIndex] = point[attr];
                        hyperBlockMaxesC[currentClassIndex] = point[attr];
                        currentClassIndex++;
                    }
                    for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                        hyperBlockMinsC[currentClassIndex] = -numeric_limits<float>::infinity();
                        hyperBlockMaxesC[currentClassIndex] = numeric_limits<float>::infinity();
                        currentClassIndex++;
                    }
                }
            }
        }

        // Process other class points
        int otherClassIndex = 0;
        for (int currentClass = 0; currentClass < allData.size(); currentClass++) {
            if (currentClass == classN) continue;

            for (const auto& point : allData[currentClass]) {
                for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                    pointsC[otherClassIndex++] = point[attr];
                }
                for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                    pointsC[otherClassIndex++] = -numeric_limits<float>::infinity();
                }
            }
        }

        // Add the existing blocks from intervalHyper
        for (auto it = hyperBlocks.begin(); it != hyperBlocks.end(); ++it) {
            if (it->classNum == classN) {
                for (int i = 0; i < it->minimums.size(); i++) {
                    //if (removed[i]) continue;
                    hyperBlockMinsC[currentClassIndex] = it->minimums[i][0];
                    hyperBlockMaxesC[currentClassIndex] = it->maximums[i][0];
                    currentClassIndex++;
                }
                for (int leftOverAtt = FIELD_LENGTH; leftOverAtt < PADDED_LENGTH; leftOverAtt++) {
                    hyperBlockMinsC[currentClassIndex] = -numeric_limits<float>::infinity();
                    hyperBlockMaxesC[currentClassIndex] = numeric_limits<float>::infinity();
                    currentClassIndex++;
                }
            }
        }

        // Allocate device memory
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_points;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;

        cudaMalloc(&d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_deleteFlags, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));
        cudaMemset(d_deleteFlags, 0, (sizeWithoutHBpoints / PADDED_LENGTH) * sizeof(int));

        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = hyperBlockMinsC.size() / PADDED_LENGTH;
        vector<int> seedQueue(numBlocks);
        for(int i = 0; i < numBlocks; i++){
            seedQueue[i] = i;
        }

        cudaMalloc(&d_mergable, numBlocks * sizeof(int));
        cudaMemset(d_mergable, 0, numBlocks * sizeof(int));
        cudaMalloc(&d_seedQueue, numBlocks * sizeof(int));
        cudaMalloc(&d_writeSeedQueue, numBlocks * sizeof(int));

        // Copy data to device
        cudaMemcpy(d_hyperBlockMins, hyperBlockMinsC.data(), sizeWithoutHBpoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hyperBlockMaxes, hyperBlockMaxesC.data(), sizeWithoutHBpoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, pointsC.data(), pointsC.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seedQueue, seedQueue.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

        cout << "Launched a kernel for class: " << classN << endl;

        // funky wap to swap the readQueue and writeQueue
        int* queues[2] = {d_seedQueue, d_writeSeedQueue};
        for(int i = 0; i < numBlocks; i++){
            // swap between the two queues
            int* readQueue = queues[i & 1];
            int* writeQueue = queues[(i + 1) & 1];
            mergerHyperBlocksWrapper(
                i, 			// seednum
                readQueue,  // seedQueue
                numBlocks,  // number seed blocks
                PADDED_LENGTH,	// num attributes
                pointsC.size() / PADDED_LENGTH,	// num op class points
                d_points,						// op class points
                d_hyperBlockMins,				// mins
                d_hyperBlockMaxes,				// maxes
                d_deleteFlags,
                d_mergable,						// mergable flags
                gridSize,
                blockSize,
                sharedMemSize
            );
            cudaDeviceSynchronize();

            // Reorder the seedblock order
            rearrangeSeedQueueWrapper(i, readQueue, writeQueue, d_deleteFlags, d_mergable, numBlocks, gridSize, blockSize);
            cudaDeviceSynchronize();

            // Reset mergable flags
            resetMergableFlagsWrapper(d_mergable, numBlocks, gridSize, blockSize);
            cudaDeviceSynchronize();
        }

        // Copy results back
        cudaMemcpy(hyperBlockMinsC.data(), d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hyperBlockMaxesC.data(), d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(deleteFlagsC.data(), d_deleteFlags, deleteFlagsC.size() * sizeof(int), cudaMemcpyDeviceToHost);
        // Process results
        for (int i = 0; i < hyperBlockMinsC.size(); i += PADDED_LENGTH) {
            
            if (deleteFlagsC[i / PADDED_LENGTH] == -1) continue;  // -1 is a seed block which was merged to. so it doesn't need to be copied back.

            vector<vector<float>> blockMins(FIELD_LENGTH);
            vector<vector<float>> blockMaxes(FIELD_LENGTH);
            for (int j = 0; j < FIELD_LENGTH; j++) {
                blockMins[j].push_back(hyperBlockMinsC[i + j]);
                blockMaxes[j].push_back(hyperBlockMaxesC[i + j]);
            }
            HyperBlock hb(blockMaxes, blockMins, classN);
            resultingBlocks[classN].emplace_back(hb);
        }

        // Free device memory
        cudaFree(d_hyperBlockMins);
        cudaFree(d_hyperBlockMaxes);
        cudaFree(d_deleteFlags);
        cudaFree(d_points);
        cudaFree(d_mergable);
        cudaFree(d_seedQueue);
        cudaFree(d_writeSeedQueue);
    }

    hyperBlocks.clear();
    for(const vector<HyperBlock>& classBlocks : resultingBlocks) {
      hyperBlocks.insert(hyperBlocks.end(), classBlocks.begin(), classBlocks.end());
    }
}


/**
* This will save the normalized dataset back so that we can use the same one in DV with the same normalization.
*/
void saveNormalizedVersionToCsv(string fileName, vector<vector<vector<float>>>& data) {
    ofstream outFile(fileName);

    if (!outFile.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return;
    }

    // Assuming all classes have at least one point, get feature count from the first point of the first class
    int featureCount = data[0][0].size();

    // Write the header
    for (int i = 0; i < featureCount; i++) {
        outFile << "x" << i << ",";
    }
    outFile << "label\n";  // Add label column

    // Iterate through classes
    for (int i = 0; i < data.size(); i++) {
        // Iterate through points in class
        for (int j = 0; j < data[i].size(); j++) {
            // Iterate through attributes of a point
            for (int k = 0; k < data[i][j].size(); k++) {
                outFile << data[i][j][k] << ",";
            }
            outFile << i << "\n";  // Append class label
        }
    }

    outFile.close();
}


/**
* We generate a confusion matrix, but allow for points to fall into multiple blocks at a time
* that is why we go through blocks on outerloop and whole dataset on the inside.
*/
vector<vector<long>> testAccuracyOfHyperBlocks(vector<HyperBlock>& hyperBlocks, vector<vector<vector<float>>> testSet){

  	// Keep track of which points were never inside of a block, when a point is classifed we increment the map internal vectors correct positon
    // there should be CLASS_NUM unordered_maps or just hashmaps, in each will hold a vector<point_index, vector<int> of len(class_num)>
    vector<unordered_map<int, vector<int>>> pointsNotClassified(CLASS_MAP.size());

    // Go through each class
    for(int cls = 0; cls < NUM_CLASSES; cls++){
        // Put the index of each point in each class into a set, this is how we will track which points were never classified.
        for(int j = 0; j < testSet[cls].size(); j++){
            pointsNotClassified[cls][j] = vector<int>(NUM_CLASSES);
        }
    }


	// Make a n x n matrix for the confusion matrix
	vector<vector<long>> ultraConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));
    vector<vector<long>> regularConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));


    cout << "Testing on " << hyperBlocks.size() << " hyperblocks" << endl;
    cout << "Testing on " << testSet.size() << " classes" << endl;
    cout << "Testing on " << testSet[0].size() << " points in first class." << endl;
    cout << "Testing on " << NUM_CLASSES << " classes" << endl;
    cout << "Testing on " << FIELD_LENGTH << " attributes" << endl;

    bool anyPointWasInside = false;

    // Go through all the blocks
	for(int hb = 0; hb < hyperBlocks.size(); hb++){
        HyperBlock& currBlock = hyperBlocks[hb];
        // Go through all the classes in the testSet
		for(int cls = 0; cls < NUM_CLASSES; cls++){
            // go through all the points in a clases
        	for(int pnt = 0; pnt < testSet[cls].size(); pnt++){
           		const vector<float>& point = testSet[cls][pnt];

                if(currBlock.inside_HB(point.size(), point.data())){

					ultraConfusionMatrix[cls][currBlock.classNum]++;


                    // Go to the actual class, to the right points entry, and increment the "predicted" class (the hb it was in).
                    pointsNotClassified[cls][pnt][currBlock.classNum]++;
                }
        	}
     	}
    }

    for(int i = 0; i < NUM_CLASSES; i++){
        cout << pointsNotClassified[0][0][i] << endl;
    }

    // Lets count how many points fell into blocks of multiple classes
    for(int i = 0; i < NUM_CLASSES; i++){
       int numPointsInMultipleClasses = 0;
       int numPointsInNoBlocks = 0;

       // Go through all the points in a class.
       for(int pnt = 0; pnt < testSet[i].size(); pnt++){
           char in = 0;

           // Go through the classification vector for the point
           for(int cls = 0; cls < NUM_CLASSES; cls++){
               if(pointsNotClassified[i][pnt][cls] > 0){
                  in++;
               }

               // Means it fell into multiple of the same.
               if(in > 1){
                   break;
               }
           }

           if(in > 1) numPointsInMultipleClasses++;

           if(in == 0) numPointsInNoBlocks++;
       }

       cout << "CLASS: " << CLASS_MAP_INT[i] << "NUM POINTS IN MULTIPLE CLASSES BLOCKS: " << numPointsInMultipleClasses << endl;
       cout << "CLASS: " << CLASS_MAP_INT[i] << "NUM POINTS IN NO BLOCKS: " << numPointsInNoBlocks << endl;
    }




    vector<vector<vector<float>>> unclassifiedPointVec(NUM_CLASSES, vector<vector<float>>()); // [class][pointIdx][attr]

    // Lets count how many points fell into blocks of multiple classes
    for(int i = 0; i < NUM_CLASSES; i++){
        // Go through all the points in a class.
        for(int pnt = 0; pnt < testSet[i].size(); pnt++){
            int majorityClass = -1;
            int max = 0;

            // Go through the classification vector for the point
            for(int cls = 0; cls < NUM_CLASSES; cls++){
                if(pointsNotClassified[i][pnt][cls] > max){
                   max = pointsNotClassified[i][pnt][cls];
                   majorityClass = cls;
                }
            }

            // The majority was the one they are actually predicted to be in
            if(majorityClass != -1){
                regularConfusionMatrix[i][majorityClass]++;

            }else{
                // Put the point that wasn't classified into the vector to go to Knn
                unclassifiedPointVec[i].push_back(testSet[i][pnt]);
            }
        }
    }

    cout << "\n\n\n\n" << endl;
    cout << "============================ REGULAR CONFUSION MATRIX ==================" << endl;
    printConfusionMatrix(regularConfusionMatrix);
    cout << "============================ END CONFUSION MATRIX ======================" << endl;

	cout << "Any point was inside" << anyPointWasInside <<  endl;

    cout << "\n\n\n\n" << endl;
    cout << "============================ K-NN CONFUSION MATRIX ==================" << endl;
    vector<vector<long>> secondConfusionMatrix = kNN(unclassifiedPointVec, hyperBlocks, 5);
    printConfusionMatrix(secondConfusionMatrix);
    cout << "============================ END K-NN MATRIX ======================" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            regularConfusionMatrix[i][j] = regularConfusionMatrix[i][j] + secondConfusionMatrix[i][j];
        }
    }

    cout << "\n\n\n\n" << endl;
    cout << "============================ DISTINCT POINT CONFUSION MATRIX ==================" << endl;
    printConfusionMatrix(regularConfusionMatrix);
    cout << "============================ END DISTINCE POINT MATRIX ======================" << endl;
    cout << "\n\n\n\n" << endl;

    return ultraConfusionMatrix;
}


void printConfusionMatrix(vector<vector<long>>& data) {
    vector<string> classLabels(NUM_CLASSES);

    vector<float> accuracies(NUM_CLASSES, 0.0);

    // Calculate the accuracies of each of the rows.
    // Only the diagonal values are correct predictions
    long overallCorrect = 0;
    long overallIncorrect = 0;
    long overallTotalClassifications = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        long correct = 0;
        long incorrect = 0;
        long totalClassifications = 0;

        for (int j = 0; j < NUM_CLASSES; ++j) {
            totalClassifications += data[i][j];
            if (i == j) {
                correct += data[i][j];  // Diagonal value indicates correct predictions
            } else {
                incorrect += data[i][j];  // Off-diagonal values are incorrect predictions
            }
        }

        if (totalClassifications > 0) {
            accuracies[i] = (float)correct / totalClassifications;
        }

        overallCorrect += correct;
        overallIncorrect += incorrect;
        overallTotalClassifications += totalClassifications;
    }

    // Overall Accuracy, prevent divide by 0 with the ternary
    float overallAccuracy = (overallTotalClassifications != 0) ? ((float)overallCorrect / overallTotalClassifications) : 0;

    // Calculate column width based on the longest class name and largest number
    size_t maxWidth = 8; // Minimum width

    for (const auto& name : classLabels) {
        maxWidth = max(maxWidth, name.length() + 2);
    }

    for (const auto& row : data) {
        for (const auto& cell : row) {
            string numStr = to_string(cell);
            maxWidth = max(maxWidth, numStr.length() + 2);
        }
    }

    // Print header row with "Actual\Predicted" in the corner
    cout << setw(maxWidth) << "Act\\Pred" << " |";
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << setw(maxWidth) << CLASS_MAP_INT[i] << " |";
    }
    cout << endl;

    // Print separator line
    cout << string(maxWidth, '-') << "-+";
    for (size_t i = 0; i < CLASS_MAP_INT.size(); i++) {
        cout << string(maxWidth, '-') << "-+";
    }
    cout << endl;

    // Print each row with row label
    for (size_t i = 0; i < data.size(); i++) {
        cout << setw(maxWidth) << CLASS_MAP_INT[i] << " |";

        for (size_t j = 0; j < data[i].size(); j++) {
            cout << setw(maxWidth) << data[i][j] << " |";
        }

        cout << accuracies[i] << endl;
    }

    cout << "The overall accuracy is " << overallAccuracy << endl;
}

// Function to clear the console screen (cross-platform)
void clearScreen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

#ifdef _WIN32
    const string PATH_SEPARATOR = "\\";
#else
    const string PATH_SEPARATOR = "/";
#endif

// Function to wait for user input before continuing
void waitForEnter() {
    cout << "\nPress Enter to continue...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

// Function to display the main menu
void displayMainMenu() {
    clearScreen();
    cout << "=== HyperBlock Classification System ===\n\n";
    cout << "1. Import training data.\n";
    cout << "2. Import testing data.\n";
    cout << "3. Save normalized training data.\n";
    cout << endl;
    cout << "4. Import existing hyperblocks.\n";
    cout << "5. Export existing hyperblocks.\n";
    cout << "6. Generate new hyperblocks.\n";
    cout << "7. Simplify hyperblocks.\n";
    cout << "8. Test hyperblocks on dataset.\n";
    cout << endl;
    cout << "9. Exit\n\n";
}

/**
*    This is the function we will use to classify data that was outside the bounds of all hyperBlocks
*
*    We will take a point and find its K Nearest Neigbors and then use a simple voting majority of these
*    to assign the point to the correct class.
*
*/
vector<vector<long>> kNN(vector<vector<vector<float>>> unclassifiedData, vector<HyperBlock>& hyperBlocks, int k){

    if(k > hyperBlocks.size()) k = (int) sqrt(hyperBlocks.size());


    // Keep track of assignments with something
    vector<vector<float>> classifications(NUM_CLASSES);    // [class][pointIndex]
    for(int i = 0; i < NUM_CLASSES; i++){
      classifications[i] = vector<float>(unclassifiedData[i].size());    // Put the vector for each class
    }

    // Flatten out the hyperBlocks into their centers
    vector<vector<vector<float>>> hyperBlockCentroids(NUM_CLASSES);    //[class][block][attribute]

    for(const auto& hyperBlock : hyperBlocks){
        // Get the center of the block
        vector<float> blockCenter(FIELD_LENGTH, 0);
        for(int i = 0; i < FIELD_LENGTH; i++){
            blockCenter[i] = (hyperBlock.maximums[i][0] + hyperBlock.minimums[i][0]) / 2.0f;
        }

        hyperBlockCentroids[hyperBlock.classNum].push_back(blockCenter);
    }




    // For each class of points
    for(int i = 0; i < NUM_CLASSES; i++){

        // For each point in unclassified points
        for(int point = 0; point < unclassifiedData[i].size(); point++){
            // Use a priority queue to keep track of the top k best distances
            priority_queue<pair<float, int>> kNearest;


            // Go through all the blocks and find the distances to their centers
            for(int blockClass = 0; blockClass < NUM_CLASSES; blockClass++){
                for(const auto& currHBCenter : hyperBlockCentroids[blockClass]){
                    // Find the distance between the HB center and the unclassified data point
                    float distance = euclideanDistance(currHBCenter, unclassifiedData[i][point]);

                    if(kNearest.size() < k){    // always add when queue is not at k yet.
                        kNearest.push(make_pair(distance, blockClass));
                    }
                    else if(distance < kNearest.top().first){ // Queue is big enough, and this distance is better than the worst in queue
                        kNearest.pop();    // pop the max (worst distance)
                        kNearest.push(make_pair(distance, blockClass));    // push the better distance.
                    }
                }
            }

            // Count votes for each class
            vector<int> votes(NUM_CLASSES, 0);
            while(!kNearest.empty()){
                votes[kNearest.top().second]++;
                kNearest.pop();
            }

            cout << "real class: " << i << " :  ";
            for(const auto vote: votes){
               cout << vote << ", ";
            }
            cout << endl;
            int majorityClass = 5;
            int maxVotes = 0;

            for(int c = 0; c < NUM_CLASSES; c++){
                if(votes[c] > maxVotes){
                   maxVotes = votes[c];
                   majorityClass = c;
                }
            }

            // WE WILL ASSUME WE DONT HAVE A ID COLUMN.
            // WE WILL ASSSUME THE LAST COLUMN IS A CLASS COLUMN
            classifications[i][point] = majorityClass;
        }
    }

    vector<vector<long>> regularConfusionMatrix(NUM_CLASSES, vector<long>(NUM_CLASSES, 0));

    // Go through the classes.
    for(int classN = 0; classN < NUM_CLASSES; classN++){
        for(int point = 0; point < classifications[classN].size(); point++){
            regularConfusionMatrix[classN][classifications[classN][point]]++;
        }
    }

    return regularConfusionMatrix;
}

vector<vector<float>> flattenMinsMaxesForRUB(vector<HyperBlock>& hyper_blocks){
    // Declare our vectors not using the "vexing declaration"
    vector<float> flatMinsList;
    vector<float> flatMaxesList;
    vector<float> blockEdges;
    vector<float> blockClasses(hyper_blocks.size());

    // First block starts at index 0
    blockEdges.push_back(0.0f);

    // Iterate over each hyper block
    for (size_t hb = 0; hb < hyper_blocks.size(); hb++) {
        // Use a reference so we don't copy the block
        const HyperBlock &block = hyper_blocks[hb];
        blockClasses[hb] = static_cast<float>(block.classNum);

        int blockLength = 0;
        // Process each attribute in this block
        for (size_t m = 0; m < block.minimums.size(); m++) {
            // The number of intervals for the attribute
            float numIntervals = static_cast<float>(block.minimums[m].size());
            blockLength += block.minimums[m].size();
            // Push the number of intervals first, as in the Java version
            flatMinsList.push_back(numIntervals);
            flatMaxesList.push_back(numIntervals);
            // Use the built-in insert method to add the interval values
            flatMinsList.insert(flatMinsList.end(), block.minimums[m].begin(), block.minimums[m].end());
            flatMaxesList.insert(flatMaxesList.end(), block.maximums[m].begin(), block.maximums[m].end());
        }
        // Add the cumulative length of this block (plus FIELD_LENGTH offset, like DV.fieldLength in Java)
        blockEdges.push_back(blockEdges.back() + blockLength + FIELD_LENGTH);
    }

    // Assemble the result using move semantics to avoid extra copies
    vector<vector<float>> result;
    result.push_back(move(flatMinsList));
    result.push_back(move(flatMaxesList));
    result.push_back(move(blockEdges));
    result.push_back(move(blockClasses));
    return result;
}

vector<vector<float>> flattenDataset(vector<vector<vector<float>>>& data) {
    vector<float> dataset;
    vector<float> classBorder(data.size() + 1);
    classBorder[0] = 0.0f;

    // For each class
    for (size_t classN = 0; classN < data.size(); ++classN) {
        // Set the end index of the class in the flattened dataset.
        // (data[classN].size() returns the number of points in that class.)
        classBorder[classN + 1] = static_cast<float>(data[classN].size()) + classBorder[classN];

        // For each point in this class, append its attributes to the dataset.
        for (const auto& point : data[classN]) {
            // For each attribute in the point
            for (const auto& attribute : point) {
                dataset.push_back(attribute);
            }
        }
    }
    vector<vector<float>> result;
    result.push_back(move(dataset));
    result.push_back(move(classBorder));
    return result;
}

// runs our three kernel functions which remove useless blocks.
void removeUselessBlocks(vector<vector<vector<float>>> &data, vector<HyperBlock>& hyper_blocks) {
    /*
     * The algorithm to remove useless blocks does basically this.
     *     - take one particular point in our dataset. Find the first HB that it fits into.
     *     - then, once everyone has found their first choice, we sum up the count of which HBs have how many points
     *     - then, we run it again, this time starting from the block which each point chose. We pick a new HB instead, if we find one which our point falls into, and which has a higher amount of points in it than our current
     *     - this is not a perfect way of doing it, but at least allows us to find the "most general blocks" based on the count of how many points are in each. This way we can then just delete whichever blocks we find with no *UNIQUE* points in them.
     *     * notice how we are putting all data in, and all blocks together. this allows us to find errors as well. we may find that a block is letting in wrong class points this way.
     */

    vector<vector<float>> minMaxResult = flattenMinsMaxesForRUB(hyper_blocks);
    vector<vector<float>> flattenedData = flattenDataset(data);

    // Use references to avoid copying.
    const vector<float>& blockMins   = minMaxResult[0];
    const vector<float>& blockMaxes  = minMaxResult[1];

    // Cast each element from the third vector (floats) into ints.
    const vector<float> &edgesAsFloats = minMaxResult[2];
    vector<int> blockEdges;
    blockEdges.resize(minMaxResult[2].size());
    // cast result [2] to ints, since this is the block edges. the array which tells us where each block starts and ends (as indexes).
    transform(edgesAsFloats.begin(), edgesAsFloats.end(), blockEdges.begin(),
              [](float val) -> int { return static_cast<int>(val); });

    // Get the dataPointsArray (again using a reference).
    const vector<float>& dataPointsArray = flattenedData[0];

    const int numPoints = dataPointsArray.size() / FIELD_LENGTH;
    vector<int> dataPointBlocks(numPoints, 0);              // Each point's chosen block.
    const int numBlocks = hyper_blocks.size();                    // Number of hyperblocks.
    vector<int> numPointsInBlocks(numBlocks, 0);              // Count of points in each hyperblock.

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

    // Remove hyperblocks that have no unique points.
    for (int i = numPointsInBlocks.size() - 1; i >= 0; i--) {
        if (numPointsInBlocks[i] == 0)
            hyper_blocks.erase(hyper_blocks.begin() + i);
    }
}

// our function to flatten our list of HBs without encoding lengths in. this is what we use for removing attirbutes
vector<vector<float>> flatMinMaxNoEncode(vector<HyperBlock> hyper_blocks) {

    int size = hyper_blocks.size();
    vector<float> flatMinsList;
    vector<float> flatMaxesList;
    vector<float> blockEdges(size + 1, 0.0f);
    vector<float> blockClasses(size, 0.0f);
    vector<float> intervalCounts(size * FIELD_LENGTH, 0.0f);

    // First block starts at 0.
    blockEdges[0] = 0.0f;
    int idx = 0;

    // Process each hyper block
    for (size_t hb = 0; hb < size; hb++) {
        HyperBlock& block = hyper_blocks[hb];
        // cast the classNum as a float so that we can put it in the vector of floats we are returning
        blockClasses[hb] = static_cast<float>(block.classNum);
        int length = 0;

        // Iterate through each attribute in the block
        for (size_t m = 0; m < block.minimums.size(); m++) {
            // Number of possible MIN/MAX values for the attribute
            size_t numIntervals = block.minimums[m].size();
            length += static_cast<int>(numIntervals);

            // Record the count of intervals for the current attribute
            intervalCounts[idx] = static_cast<float>(numIntervals);
            idx++;

            // Add all intervals for the current attribute.
            for (size_t i = 0; i < numIntervals; i++) {
                flatMinsList.push_back(block.minimums[m][i]);
                flatMaxesList.push_back(block.maximums[m][i]);
            }
        }
        // Mark the end of the block by accumulating the length.
        blockEdges[hb + 1] = blockEdges[hb] + length;
    }

    // Return the five arrays in a vector (order matches the original Java return)
    return { flatMinsList, flatMaxesList, blockEdges, blockClasses, intervalCounts };
}

void removeUselessAttributesCUDA(vector<HyperBlock> &hyper_blocks, vector<vector<vector<float>>> & data, vector<vector<int>> &attributeOrderings) {
    // Prepare host data by flattening your data structures.
    auto fMinMaxResult = flatMinMaxNoEncode(hyper_blocks);
    auto fDataResult = flattenDataset(data);

    // Build host arrays from the flattened results:
    vector<float> mins = fMinMaxResult[0];
    vector<float> maxes = fMinMaxResult[1];
    int minMaxLen = static_cast<int>(mins.size());

    vector<int> blockEdges(fMinMaxResult[2].size());
    for (size_t i = 0; i < fMinMaxResult[2].size(); i++) {
        blockEdges[i] = static_cast<int>(fMinMaxResult[2][i]);
    }
    int numBlocks = static_cast<int>(hyper_blocks.size());

    vector<int> blockClasses(fMinMaxResult[3].size());
    for (size_t i = 0; i < fMinMaxResult[3].size(); i++) {
        blockClasses[i] = static_cast<int>(fMinMaxResult[3][i]);
    }

    vector<int> intervalCounts(fMinMaxResult[4].size());
    for (size_t i = 0; i < fMinMaxResult[4].size(); i++) {
        intervalCounts[i] = static_cast<int>(fMinMaxResult[4][i]);
    }

    // Create flags array (initialize to 0).
    vector<char> attrRemoveFlags(hyper_blocks.size() * FIELD_LENGTH, 0);

    // Prepare the dataset.
    vector<float> dataset = fDataResult[0];
    int numPoints = static_cast<int>(dataset.size() / FIELD_LENGTH);

    vector<int> classBorder(fDataResult[1].size());
    for (size_t i = 0; i < fDataResult[1].size(); i++) {
        classBorder[i] = static_cast<int>(fDataResult[1][i]);
    }
    int numClasses = static_cast<int>(hyper_blocks.size());

    vector<int> attributeOrderingsFlattened(attributeOrderings.size() * FIELD_LENGTH, 0);
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

/**
 *     EUCLIDEAN DISTANCE OF TWO VECTORS.
 */
float euclideanDistance(const vector<float>& hbCenter, const vector<float>& point){
    float sumSquaredDifference = 0.0f;

    for(int i = 0; i < FIELD_LENGTH; i++){
        float diff = hbCenter[i] - point[i];
        sumSquaredDifference += diff * diff;
    }

    return sqrt(sumSquaredDifference);
}

vector<int> runSimplifications(vector<HyperBlock> &hyperBlocks, vector<vector<vector<float>>> &trainData, vector<vector<int>> &bestAttributeOrderings){

    int runCount = 0;
    int totalClauses = 0;
    int updatedClauses = 0;

    do{
        // set our count of what we have to start
        totalClauses = updatedClauses;
        runCount++; // counter so we can show how many iterations this took.

        // simplification functions
        removeUselessAttributesCUDA(hyperBlocks, trainData, bestAttributeOrderings);
        removeUselessBlocks(trainData, hyperBlocks);

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

    // iteratively call the simplifications until we don't remove any more clauses.
    } while(updatedClauses != totalClauses);
    return { runCount, totalClauses };
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
    trainingData = dataSetup(argv[1], CLASS_MAP, CLASS_MAP_INT);
    cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << endl;
    cout << "NUM CLASSES : " << NUM_CLASSES << endl;

    // Resize normalization vectors based on FIELD_LENGTH
    minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
    maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());

    findMinMaxValuesInDataset(trainingData, minValues, maxValues);
    minMaxNormalization(trainingData, minValues, maxValues);

    // Run LDA on the training data.
    vector<vector<float>>bestVectors = linearDiscriminantAnalysis(trainingData);

    // Initialize indexes for each class
    vector<vector<int> > bestVectorsIndexes = vector<vector<int> >(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
    vector<int> eachClassBestVectorIndex = vector<int>(NUM_CLASSES);

    // sort our vectors from the LDA by their coefficients so that we can determine an ordering for removing and sorting by best columns in generation
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < FIELD_LENGTH; j++) {
            bestVectorsIndexes[i][j] = j;
        }
        // Sort indices by absolute value of the coefficients for the current class.
        sort(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
             [&](int a, int b) {
                 return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
             });
        eachClassBestVectorIndex[i] = bestVectorsIndexes[i][0];
    }

    generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex);
    cout << "HYPERBLOCK GENERATION FINISHED!" << endl;
    cout << "WE FOUND " << hyperBlocks.size() << " HYPERBLOCKS!" << endl;

    vector<int> result = runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
    int totalPoints = 0;
    for (const auto &c : trainingData)
        totalPoints += c.size();
    cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
    cout << "Ran simplifications: " << result[0] << " Times" << endl;
    cout << "We had: " << totalPoints << " points\n";

    saveBasicHBsToCSV(hyperBlocks, "AsyncBlockOutput");
    return 0;
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

    vector<float> minValues;
    vector<float> maxValues;

    vector<HyperBlock> hyperBlocks;

    vector<vector<long>> ultraConfusionMatrix;

    vector<vector<float>> bestVectors;
    vector<vector<int>> bestVectorsIndexes;
    vector<int> eachClassBestVectorIndex;

    bool running = true;
    int choice;

    while (running) {
        displayMainMenu();
        cin >> choice;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        switch (choice) {
            case 1: { // IMPORT TRAINING DATA
                cout << "Enter training data filename: " << endl;
                system("ls datasets");  // list available datasets
                getline(cin, trainingDataFileName);
                // Prepend the directory (adjust PATH_SEPARATOR as needed)
                string fullPath = "datasets" + string(PATH_SEPARATOR) + trainingDataFileName;
                trainingData = dataSetup(fullPath.c_str(), CLASS_MAP, CLASS_MAP_INT);

                // Resize normalization vectors based on FIELD_LENGTH
                minValues.assign(FIELD_LENGTH, numeric_limits<float>::infinity());
                maxValues.assign(FIELD_LENGTH, -numeric_limits<float>::infinity());
                findMinMaxValuesInDataset(trainingData, minValues, maxValues);
                minMaxNormalization(trainingData, minValues, maxValues);

                // Run LDA on the training data.
                bestVectors = linearDiscriminantAnalysis(trainingData);

                bestVectorsIndexes = vector<vector<int>>(NUM_CLASSES, vector<int>(FIELD_LENGTH, 0));
                eachClassBestVectorIndex = vector<int>(NUM_CLASSES);

                for (int i = 0; i < NUM_CLASSES; i++) {
                    for (int j = 0; j < FIELD_LENGTH; j++) {
                        bestVectorsIndexes[i][j] = j;
                    }
                    sort(bestVectorsIndexes[i].begin(), bestVectorsIndexes[i].end(),
                         [&](int a, int b) {
                             return fabs(bestVectors[i][a]) < fabs(bestVectors[i][b]);
                         });
                    eachClassBestVectorIndex[i] = bestVectorsIndexes[i][0];
                }
                waitForEnter();
                break;
            }
            case 2: { // IMPORT TESTING DATA
                cout << "Enter testing data filename: " << endl;

                system("ls datasets");  // list available datasets
                getline(cin, testingDataFileName);
                // Prepend the directory (adjust PATH_SEPARATOR as needed)
                string fullPath = "datasets" + string(PATH_SEPARATOR) + testingDataFileName;

                testData = dataSetup(fullPath, CLASS_MAP_TESTING, CLASS_MAP_TESTING_INT);
                // Normalize and reorder testing data as needed.
                normalizeTestSet(testData, minValues, maxValues);
                testData = reorderTestingDataset(testData, CLASS_MAP, CLASS_MAP_TESTING);
                waitForEnter();
                break;
            }
            case 3: { // SAVE NORMALIZED TRAINING DATA
                cout << "Enter the file to save the normalized training data to: " << endl;
                getline(cin, normalizedSaveFile);
                // saveNormalizedVersionToCsv(normalizedSaveFile, trainingData);
                cout << "Saved normalized training data to: " << normalizedSaveFile << endl;
                waitForEnter();
                break;
            }
            case 4: { // IMPORT EXISTING HYPERBLOCKS
                cout << "Enter existing hyperblocks file name: " << endl;
                getline(cin, hyperBlocksImportFileName);
                // hyperBlocks = loadBasicHBsFromCSV(hyperBlocksImportFileName);
                cout << "HyperBlocks imported from file " << hyperBlocksImportFileName << " successfully" << endl;
                waitForEnter();
                break;
            }
            case 5: { // EXPORT HYPERBLOCKS
                cout << "Enter the file to save HyperBlocks to: " << endl;
                getline(cin, hyperBlocksExportFileName);
                // saveBasicHBsToCSV(hyperBlocks, hyperBlocksExportFileName);
                break;
            }
            case 6: { // GENERATE NEW HYPERBLOCKS
                if (trainingData.empty()) {
                    cout << "\nError: Please import training data first." << endl;
                    waitForEnter();
                } else {
                    hyperBlocks.clear();
                    generateHBs(trainingData, hyperBlocks, eachClassBestVectorIndex);
                }
                cout << "Finished Generating HyperBlocks" << endl;
                waitForEnter();
                break;
            }
            case 7: { // SIMPLIFY HYPERBLOCKS
                vector<int> result = runSimplifications(hyperBlocks, trainingData, bestVectorsIndexes);
                int totalPoints = 0;
                for (const auto &c : trainingData)
                    totalPoints += c.size();
                cout << "After removing useless blocks we have: " << result[1] << " clauses\n";
                cout << "Ran simplifications: " << result[0] << " Times" << endl;
                cout << "We had: " << totalPoints << " points\n";
                waitForEnter();
                break;
            }
            case 8: { // TEST HYPERBLOCKS ON DATASET
                cout << "Testing hyperblocks on testing dataset" << endl;
                ultraConfusionMatrix = testAccuracyOfHyperBlocks(hyperBlocks, testData);
                printConfusionMatrix(ultraConfusionMatrix);
                waitForEnter();
                break;
            }
            case 9: {
                running = false;
                break;
            }
            default: {
                cout << "\nInvalid choice. Please try again." << endl;
                waitForEnter();
                break;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Main entry point: choose mode based on argc.
int main(int argc, char* argv[]) {

    // asynchronous mode. useful for giant datasets that are going to take hours.
    if (argc >= 2)
        return runAsync(argc, argv);

    // this is for testing results and testing small data.
    runInteractive();
    return 0;
}