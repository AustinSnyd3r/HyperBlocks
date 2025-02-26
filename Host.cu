#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <future>
#include "CudaUtil.h"
#include "HyperBlock.h"
#include "HyperBlockCuda.cuh"
#include <chrono>
#include <omp.h>
using namespace std;

int NUM_CLASSES;   // Number of classes in the dataset
int NUM_POINTS;    // Total number of points in the dataset
int FIELD_LENGTH;  // Number of attributes in the dataset

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

Interval longest_interval(vector<DataATTR>& data_by_attr, float acc_threshold, vector<HyperBlock>& existing_hb, int attr);
void remove_value_from_interval(vector<DataATTR>& data_by_attr, Interval& intr, float value);
int skip_value_in_interval(vector<DataATTR>& data_by_attr, int i, float value);
bool check_interval_overlap(vector<DataATTR>& data_by_attr, Interval& intr, int attr, vector<HyperBlock>& existing_hb);
void merger_cuda(const vector<vector<vector<float>>>& data_with_skips, const vector<vector<vector<float>>>& all_data, vector<HyperBlock>& hyper_blocks);
void saveBasicHBsToCSV(const vector<HyperBlock>& hyper_blocks);
void print3DVector(const vector<vector<vector<float>>>& vec);


/***
* We want to go through the hyper_blocks that were generated and write them to a file.
*
*
* This print isn't caring about disjunctive blocks.
*/
void saveBasicHBsToCSV(const vector<HyperBlock>& hyper_blocks, const string& file_name){
	// Open file for writing
    ofstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return;
    }

	// min1, min2, min3, ..., minN, max1, max2, max3, ..., maxN, class
	for (const auto& hyper_block : hyper_blocks) {
        // Write minimums
        for (const vector<float>& min : hyper_block.minimums) {
            file << min[0] << ",";
        }

        // Write maximums
        for (const vector<float>& max : hyper_block.maximums) {
            file << max[0] << ",";
        }

        // Write the class number
        file << hyper_block.classNum << "\n";
    }

    file.close();
}

///////////////////////// FUNCTIONS FOR INTERVAL_HYPER IMPLEMENTATION /////////////////////////

 /**
     * Finds largest interval across all dimensions of a set of data.
     * @param data_by_attr all data split by attribute
     * @param acc_threshold accuracy threshold for interval
     * @param existing_hb existing hyperblocks to check for overlap
     * @return largest interval
     */
vector<DataATTR> interval_hyper(vector<vector<DataATTR>>& data_by_attr, float acc_threshold, vector<HyperBlock>& existing_hb){
    //cout << "Starting interval hyperblock" << endl;
    vector<future<Interval>> intervals;
    int attr = -1;
    Interval best(-1, -1, -1, -1);

   // Search each attribute
    for (int i = 0; i < data_by_attr.size(); i++) {
        // Launch async task
        intervals.emplace_back(async(launch::async, longest_interval, ref(data_by_attr[i]), acc_threshold, ref(existing_hb), i));
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
          	//cout << "Data by attribute printing time: " << data_by_attr[attr][i].classNum << " " << data_by_attr[attr][i].classIndex << "\n";
            //cout << attr << endl;

            longest.push_back(data_by_attr[attr][i]);
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
 * @param data_by_attr sorted data by attribute
 * @param acc_threshold accuracy threshold for interval
 * @param existing_hb existing hyperblocks to check for overlap
 * @param attr attribute to find interval on
 * @return longest interval
*/
Interval longest_interval(vector<DataATTR>& data_by_attr, float acc_threshold, vector<HyperBlock>& existing_hb, int attr){
    //cout << "Started longest interval \n" << endl;

    Interval intr(1, 0, 0, attr);
    Interval max_intr(-1, -1, -1, attr);

    int n = data_by_attr.size();
    float misclassified = 0;

    for(int i = 1; i < n; i++){
        // If current class matches with next
        if(data_by_attr[intr.start].classNum == data_by_attr[i].classNum){
            intr.size++;
        }
        else if( (misclassified+1) / intr.size > acc_threshold){
            // ^ i think this is a poor way to check. but not changing rn for the translation from java
            misclassified++;
            intr.size++;
        }
        else{
            // Remove value from interval if accuracy is below threshold.
            if(data_by_attr[i-1].value == data_by_attr[i].value){
                // remove then skip overlapped values
                remove_value_from_interval(data_by_attr, intr, data_by_attr[i].value);
                i = skip_value_in_interval(data_by_attr, i, data_by_attr[i].value);
            }

            // Update longest interval if it doesn't overlap
            if(intr.size > max_intr.size && check_interval_overlap(data_by_attr, intr, attr, existing_hb)){
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
    if(intr.size > max_intr.size && check_interval_overlap(data_by_attr, intr, attr, existing_hb)){
        max_intr.start = intr.start;
        max_intr.end = intr.end;
        max_intr.size = intr.size;
    }

    //cout << "Finished longest interval \n" << endl;

    return max_intr;
}


bool check_interval_overlap(vector<DataATTR>& data_by_attr, Interval& intr, int attr, vector<HyperBlock>& existing_hb){
    //cout << "Started check interval overlap\n" << endl;
    // interval range of vals
    float intv_min = data_by_attr[intr.start].value;
    float intv_max = data_by_attr[intr.end].value;
   
    /*
    *   check if interval range overlaps with any existing hyperblocks
    * to not overlap the interval maximum must be below all existing hyperblock minimums
    * or the interval minimum must be above all existing hyperblock maximums
    */
    for(const HyperBlock& hb : existing_hb){
        if (!(intv_max < hb.minimums.at(attr).at(0) || intv_min > hb.maximums.at(attr).at(0))){
            return false;
        }
    }

    //cout << "Finished check interval overlap\n" << endl;

    // If unique return true
    return true;
}

//skip_value_in_interval
int skip_value_in_interval(vector<DataATTR>& data_by_attr, int i, float value){
    //cout << "Starting skip value in interval\n" << endl;

    while(data_by_attr[i].value == value){
        if(i < data_by_attr.size() - 1){
            i++;
        }
        else{
            break;
        }
    }

    //cout << "Finished skip value in interval\n" << endl;

    return i;
}


//remove_value_from_interval
void remove_value_from_interval(vector<DataATTR>& data_by_attr, Interval& intr, float value){
    //cout << "Starting remove value from intervals\n" << endl;
    while(data_by_attr[intr.end].value == value){
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

///////////////////////// END FUNCTIONS FOR INTERVAL_HYPER IMPLEMENTATION /////////////////////////

void generateHBs(vector<vector<vector<float>>>& data, vector<HyperBlock>& hyper_blocks){
  	// "Started generating HBS\n" << endl;
    // Hyperblocks generated with this algorithm
    vector<HyperBlock> gen_hb;

    // Get data to create hyperblocks
    vector<vector<DataATTR>> data_by_attr = separateByAttribute(data);
    vector<vector<DataATTR>> all_intv;

    // Create dataset without data from interval HyperBlocks
    vector<vector<vector<float>>> datum;
    vector<vector<vector<float>>> seed_data;
    vector<vector<int>> skips;
	// "Initialized datum, seed_data, skips\n" << endl;

    // Initially generate blocks

        //cout << "Starting while loop to generate hyperblocks\n";
		//cout << "data_by_attr[0].size() = " << data_by_attr[0].size() << endl;

        while(data_by_attr[0].size() > 0){
			//cout << "Attempting to go into interval_hyper " << endl;

            vector<DataATTR> intv = interval_hyper(data_by_attr, 100, gen_hb);
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
                //cout << "Added results from last interval_hyper" << endl << endl;
            }else{
                //cout << "Breaking because the intv size is < 1" << endl;
                break;
            }
        }

        // Add all hbs from gen_hb to hyper_blocks
        hyper_blocks.insert(hyper_blocks.end(), gen_hb.begin(), gen_hb.end());

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
            sortByColumn(datum[i], 0);
            sortByColumn(seed_data[i], 0);
        }

    // Call CUDA function.
    //cout << "Calling merger_cuda\n" << endl;

    try{
        //cout << "Printing interval hyperblocks:\n\n" << endl;
        //for(const auto& hb : hyper_blocks){
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
		//print3DVector(datum);

        //cout << "SEED DATA BEING PASSED INTO MERGING:" << endl;
		//print3DVector(seed_data);

        merger_cuda(seed_data, datum, hyper_blocks);
    }catch (exception e){
        cout << "Error in generateHBs: merger_cuda" << endl;
    }
}




void print3DVector(const vector<vector<vector<float>>>& vec) {
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
vector<vector<vector<float>>> dataSetup(const string filepath) {
    // 3D vector: data[class][point][attribute]
    vector<vector<vector<float>>> data;

    // Map class labels to indices in `data`
    map<string, int> classMap;

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

    file.close();

    // Set global variables
    FIELD_LENGTH = data.empty() ? 0 : static_cast<int>(data[0][0].size());
    NUM_CLASSES = classNum;

    //cout << "Finished setting up data\n" << endl;

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


void minMaxNormalization(vector<vector<vector<float>>>& dataset) {
    //cout << "Starting min-max normalization\n" << endl;

    if (dataset.empty()) return;

    int num_classes = dataset.size();

    // Min and max values for each attribute
    vector<float> min_vals(FIELD_LENGTH, std::numeric_limits<float>::infinity());
    vector<float> max_vals(FIELD_LENGTH, -std::numeric_limits<float>::infinity());

    // Step 1: Find min and max for each attribute
    for (const auto& class_data : dataset) {
        for (const auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                min_vals[k] = min(min_vals[k], point[k]);
                max_vals[k] = max(max_vals[k], point[k]);
            }
        }
    }

    // Step 2: Apply Min-Max normalization
    for (auto& class_data : dataset) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                // Avoid div/0
                if (max_vals[k] != min_vals[k]) {
                    point[k] = (point[k] - min_vals[k]) / (max_vals[k] - min_vals[k]);
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
void merger_cuda(const vector<vector<vector<float>>>& data_with_skips, const vector<vector<vector<float>>>& all_data, vector<HyperBlock>& hyper_blocks) {

    // Calculate total points
    int numPoints = 0;
    for (const auto& classData : all_data) {
        numPoints += classData.size();
    }

    // Count blocks per class
    vector<int> numBlocksOfEachClass(NUM_CLASSES, 0);
    for (const auto& hb : hyper_blocks) {
        numBlocksOfEachClass[hb.classNum]++;
    }

    vector<vector<HyperBlock>> resultingBlocks(NUM_CLASSES);
    
    // get our device count, and max it so that the most we will use is numClasses if we have only 2 classes for example.
    /* MULTI GPU BUSINESS
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    deviceCount = std::min(deviceCount, NUM_CLASSES);


    // Process each class
    // we have our multithreading happen here at this level. set the device to class % deviceCount
    #pragma omp parallel for num_threads(deviceCount)
    */
    for (int classN = 0; classN < NUM_CLASSES; classN++) {
        
        // set our device based on class. this way even single threaded we use multiple GPUs
        // MORE MULTI GPU BUSINESS
        //cudaSetDevice(classN % deviceCount);
        
        int totalDataSetSizeFlat = numPoints * FIELD_LENGTH;
        int sizeWithoutHBpoints = ((data_with_skips[classN].size() + numBlocksOfEachClass[classN]) * FIELD_LENGTH);
        if (data_with_skips[classN].empty()) {
            sizeWithoutHBpoints = numBlocksOfEachClass[classN] * FIELD_LENGTH;
        }
        // Allocate host memory
        vector<float> hyperBlockMinsC(sizeWithoutHBpoints);
        vector<float> hyperBlockMaxesC(sizeWithoutHBpoints);
        vector<float> combinedMinsC(sizeWithoutHBpoints);
        vector<float> combinedMaxesC(sizeWithoutHBpoints);
        vector<int> deleteFlagsC(sizeWithoutHBpoints / FIELD_LENGTH);

        int nSize = all_data[classN].size();
        vector<float> pointsC(totalDataSetSizeFlat - (nSize * FIELD_LENGTH));

        // Fill data arrays
        int currentClassIndex = 0;
        for (int currentClass = 0; currentClass < data_with_skips.size(); currentClass++) {
            for (const auto& point : data_with_skips[currentClass]) {
                if (currentClass == classN) {
                    for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                        //if (removed[attr]) continue;
                        hyperBlockMinsC[currentClassIndex] = point[attr];
                        hyperBlockMaxesC[currentClassIndex] = point[attr];
                        currentClassIndex++;
                    }
                }
            }
        }

        // Process other class points
        int otherClassIndex = 0;
        for (int currentClass = 0; currentClass < all_data.size(); currentClass++) {
            if (currentClass == classN) continue;

            for (const auto& point : all_data[currentClass]) {
                for (int attr = 0; attr < FIELD_LENGTH; attr++) {
                    //if (removed[attr]) continue;
                    pointsC[otherClassIndex++] = point[attr];
                }
            }
        }

        // Add the existing blocks from interval_hyper
        for (auto it = hyper_blocks.begin(); it != hyper_blocks.end(); ++it) {
            if (it->classNum == classN) {
                for (int i = 0; i < it->minimums.size(); i++) {
                    //if (removed[i]) continue;
                    hyperBlockMinsC[currentClassIndex] = it->minimums[i][0];
                    hyperBlockMaxesC[currentClassIndex] = it->maximums[i][0];
                    currentClassIndex++;
                }
            }
        }

        // Allocate device memory
        float *d_hyperBlockMins, *d_hyperBlockMaxes, *d_combinedMins, *d_combinedMaxes, *d_points;
        int *d_deleteFlags, *d_mergable, *d_seedQueue, *d_writeSeedQueue;

        cudaMalloc(&d_hyperBlockMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_hyperBlockMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_combinedMins, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_combinedMaxes, sizeWithoutHBpoints * sizeof(float));
        cudaMalloc(&d_deleteFlags, (sizeWithoutHBpoints / FIELD_LENGTH) * sizeof(int));
        cudaMemset(d_deleteFlags, 0, (sizeWithoutHBpoints / FIELD_LENGTH) * sizeof(int));

        cudaMalloc(&d_points, pointsC.size() * sizeof(float));

        int numBlocks = hyperBlockMinsC.size() / FIELD_LENGTH;
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

        // Find best occupancy
        int sharedMemSize = 2 * FIELD_LENGTH * sizeof(float) + sizeof(int);
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mergerHyperBlocks, sharedMemSize, 0);

        // numblocks is how many HBs we have, and we are doing a ceiling on that so that
        int gridSize = (numBlocks + blockSize - 1) / blockSize;
        cout << "Grid size: " << gridSize << endl;
        cout << "Block size: " << blockSize << endl;

        // funky way to swap the readQueue and writeQueue
        int* queues[2] = {d_seedQueue, d_writeSeedQueue};
        for(int i = 0; i < numBlocks; i++){
            // swap between the two queues
            int* readQueue = queues[i & 1];
            int* writeQueue = queues[(i + 1) & 1];
            mergerHyperBlocksWrapper(
                i, 			// seednum
                readQueue,  // seedQueue
                numBlocks,  // number seed blocks
                FIELD_LENGTH,	// num attributes
                pointsC.size() / FIELD_LENGTH,	// num op class points
                d_points,						// op class points
                d_hyperBlockMins,				// mins
                d_hyperBlockMaxes,				// maxes
                d_deleteFlags,
                d_mergable,						// mergable flags
                gridSize,
                blockSize,
                sharedMemSize,
                d_combinedMins,
                d_combinedMaxes
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
        for (int i = 0; i < hyperBlockMinsC.size(); i += FIELD_LENGTH) {
            if (deleteFlagsC[i / FIELD_LENGTH] == -1) continue;  // -1 is a seed block which was merged to. so it doesn't need to be copied back.

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
        cudaFree(d_combinedMins);
        cudaFree(d_combinedMaxes);
        cudaFree(d_deleteFlags);
        cudaFree(d_points);
        cudaFree(d_mergable);
        cudaFree(d_seedQueue);
        cudaFree(d_writeSeedQueue);
    }

    hyper_blocks.clear();
    for(const vector<HyperBlock>& classBlocks : resultingBlocks) {
      hyper_blocks.insert(hyper_blocks.end(), classBlocks.begin(), classBlocks.end());
    }
}

// WE WILL ASSUME WE DONT HAVE A ID COLUMN.
// WE WILL ASSSUME THE LAST COLUMN IS A CLASS COLUMN
int main(int argc, char* argv[]) {
    printf("USER WARNING :: ENSURE THAT THERE IS NO ID COLUMN\n");
    printf("USER WARNING :: ENSURE THAT THE LAST COLUMN IS A CLASS COLUMN\n");

    // dimension 1 is the class, then it is just a list of points, since each point is a list of floats.
    vector<vector<vector<float>>> data = dataSetup(argv[1]);

    cout << "NUM ATTRIBUTES : " << FIELD_LENGTH << endl;
    cout << "NUM CLASSES    : " << NUM_CLASSES << endl;

    cout << "Number elements in class 0: " << data[0].size() << endl;
    cout << "Number elements in class 1: " << data[1].size() << endl;
 	// normalize the data

    minMaxNormalization(data);
	//print3DVector(data);
    printf("Made it past normalization");
    // Make the hyperblocks list to store the hyperblocks that are generated.
	vector<HyperBlock> hyper_blocks;

    // generate hyperblocks
    auto start = std::chrono::high_resolution_clock::now();

    generateHBs(data, hyper_blocks);
    auto stop = std::chrono::high_resolution_clock::now();

 	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken: " << duration_ms.count() << " ms\n";

    // Calculate duration in seconds
    auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken: " << duration_sec.count() << " s\n";
	cout << "HyperBlocks : " << hyper_blocks.size() << endl;

    cout << "Finished generating HBS\n" << endl;
    cout << "Number of HBs: " << hyper_blocks.size() << endl;

    saveBasicHBsToCSV(hyper_blocks, "testForDVinCPP.csv");
    return 0;
}
