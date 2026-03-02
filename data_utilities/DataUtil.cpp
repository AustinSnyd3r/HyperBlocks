//
// Created by Austin Snyder on 3/20/2025.
//
#include "DataUtil.h"

#include <climits>
extern int FIELD_LENGTH;
extern int NUM_CLASSES;

using namespace std;


// In-place stratified split: modifies trainingData by moving points to validationData
void DataUtil::createValidationSplit(
    vector<vector<vector<float>>>& trainingData,
    vector<vector<vector<float>>>& validationData,
    float validationFraction,
    unsigned int randomSeed
) {
    // Random for the shuffling
    mt19937 rng(randomSeed);
    validationData.clear();
    validationData.resize(trainingData.size());

    // Step 1: Find the minimum class size
    int minClassSize = INT_MAX;
    for (const auto& pointsInClass : trainingData) {
        minClassSize = min(minClassSize, static_cast<int>(pointsInClass.size()));
    }

    // Step 2: Calculate number of validation points per class
    int valPerClass = max(1, static_cast<int>(minClassSize * validationFraction));

    // Step 3: Stratified sampling with fixed count
    for (int classIdx = 0; classIdx < trainingData.size(); ++classIdx) {
        auto& pointsInClass = trainingData[classIdx];
        vector<int> indices(pointsInClass.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);

        int actualValCount = min(valPerClass, static_cast<int>(pointsInClass.size()));

        // Move to validation
        for (int i = 0; i < actualValCount; ++i) {
            validationData[classIdx].push_back(move(pointsInClass[indices[i]]));
        }

        // Remove from training (reverse sorted)
        sort(indices.begin(), indices.begin() + actualValCount, greater<int>());
        for (int i = 0; i < actualValCount; ++i) {
            pointsInClass.erase(pointsInClass.begin() + indices[i]);
        }
    }
}




/*  Returns a class seperated version of the dataset
 *  Each class has an entry in the outer vector with a 2-d vector of its points
 */
vector<vector<vector<float>>> DataUtil::dataSetup(const string filepath, map<string, int>& classMap, map<int, string>& reversedClassMap) {
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

    FIELD_LENGTH = data[0][0].size();
    NUM_CLASSES = classNum;

    return data;
}

/* This needs to be a function to serialize hyperblocks.
 * take in 3-D vector that is the hyperblocks for each class
 * each class gets a dimension, with a 2-d vector for the HBs
 * assumes each row in the 2-D vector is 1 hyperblock
 * the first 1/2 of the row is the max's, the end is the mins.
 */
void DataUtil::saveHyperBlocksToFile(const string& filepath, const vector<vector<vector<float>>>& hyperBlocks) {
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

/**
* This will save the normalized dataset back so that we can use the same one in DV with the same normalization.
*/
void DataUtil::saveNormalizedVersionToCsv(string fileName, vector<vector<vector<float>>>& data) {
    ofstream outFile(fileName);

    if (!outFile.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return;
    }

    // Assuming all classes have at least one point, get feature count from the first point of the first class
    int featureCount = FIELD_LENGTH;

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

// just loads a vector of HBs back from a file when we have exported.
vector<HyperBlock> DataUtil::loadBasicHBsFromCSV(const string& fileName) {
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
* A function to normalize the test set using the given mins/maxes that were used to normalize the initial set
*/
void DataUtil::normalizeTestSet(vector<vector<vector<float>>>& testSet, const vector<float>& minValues, const vector<float>& maxValues, int FIELD_LENGTH) {
    if (testSet.empty()){
      cout << "Test set was empty when trying to normalize" << endl;
      return;
	}


    for (auto& class_data : testSet) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                if (maxValues[k] != minValues[k]) {
                    point[k] = (point[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                } else {
                    point[k] = 0.50000f;
                }
            }
        }
    }

    for(auto& class_data : testSet){
      for (auto& point : class_data){
        for (int k = 0; k < FIELD_LENGTH; k++){
          if(point[k] > 1.0f){
	         cout << "Out of range" << point[k] << endl;
             point[k] = 1.000000f;
		  }
          if(point[k] < 0.0f){
		  	point[k] = 0.000000f;
          }
        }
      }
    }




}

// normalizes the training data using just the min and max value in each attribute
void DataUtil::minMaxNormalization(vector<vector<vector<float>>>& dataset, const vector<float>& minValues, const vector<float>& maxValues, int FIELD_LENGTH) {
    cout << "Normalizing the dataset" << endl;
    if (dataset.empty()) return;

    int num_classes = dataset.size();

    // Step 2: Apply Min-Max normalization
    for (auto& class_data : dataset) {
        for (auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                // Avoid div/0
                if (maxValues[k] != minValues[k]) {
                    point[k] = (point[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                } else {
                    //cout << "Column found with useless values" << endl;
                    point[k] = 0.500000f;
                }
            }
        }
    }
}


// Function to reorder testing dataset based on training class mapping
vector<vector<vector<float>>> DataUtil::reorderTestingDataset(const vector<vector<vector<float>>>& testingData, const map<string, int>& CLASS_MAP_TRAINING, const map<string, int>& CLASS_MAP_TESTING) {
    // Create a new vector with the same size as the testing data
    vector<vector<vector<float>>> reorderedTestingData(testingData.size());

    // Create a mapping from testing indices to training indices
    map<int, int> indexMap;
    // For each (className -> trainingIndex) in the training map
    for (const auto& entry : CLASS_MAP_TRAINING) {
        const string& className   = entry.first;
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


vector<HyperBlock> DataUtil::loadBasicHBsFromBinary(const string& fileName) {
    ifstream file(fileName, ios::binary);
    vector<HyperBlock> loadedBlocks;

    if (!file.is_open()) {
        cerr << "Error opening binary file: " << fileName << endl;
        return loadedBlocks;
    }

    int numBlocks, fieldLength;
    file.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));
    file.read(reinterpret_cast<char*>(&fieldLength), sizeof(int));

    for (int b = 0; b < numBlocks; ++b) {
        vector<vector<float>> mins(fieldLength, vector<float>(1));
        vector<vector<float>> maxs(fieldLength, vector<float>(1));

        for (int i = 0; i < fieldLength; ++i) {
            file.read(reinterpret_cast<char*>(&mins[i][0]), sizeof(float));
        }
        for (int i = 0; i < fieldLength; ++i) {
            file.read(reinterpret_cast<char*>(&maxs[i][0]), sizeof(float));
        }

        int classNum;
        file.read(reinterpret_cast<char*>(&classNum), sizeof(int));

        loadedBlocks.emplace_back(maxs, mins, classNum);
    }

    file.close();
    return loadedBlocks;
}

void DataUtil::saveBasicHBsToBinary(const vector<HyperBlock>& hyperBlocks, const string& fileName, int FIELD_LENGTH) {
    ofstream file(fileName, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening binary file: " << fileName << endl;
        return;
    }

    int numBlocks = static_cast<int>(hyperBlocks.size());
    file.write(reinterpret_cast<const char*>(&numBlocks), sizeof(int));
    file.write(reinterpret_cast<const char*>(&FIELD_LENGTH), sizeof(int)); // For sanity check during read

    for (const auto& hyperBlock : hyperBlocks) {
        // Write FIELD_LENGTH minimum values (only the first entry from each vector)
        for (int i = 0; i < FIELD_LENGTH; ++i) {
            float minVal = hyperBlock.minimums[i][0];
            file.write(reinterpret_cast<const char*>(&minVal), sizeof(float));
        }

        // Write FIELD_LENGTH maximum values
        for (int i = 0; i < FIELD_LENGTH; ++i) {
            float maxVal = hyperBlock.maximums[i][0];
            file.write(reinterpret_cast<const char*>(&maxVal), sizeof(float));
        }

        // Write class label
        file.write(reinterpret_cast<const char*>(&hyperBlock.classNum), sizeof(int));
    }

    file.close();
}

// same as regular load from binary. returns a list of HBs, but this case, it is the one to some blocks, where we have several lists of HBs since we generated many sets.
vector<vector<HyperBlock>> DataUtil::loadOneToSomeBlocksFromBinary(const string& fileName) {
    vector<HyperBlock> allBlocks = DataUtil::loadBasicHBsFromCSV(fileName);
    vector<vector<HyperBlock>> splitBlocks;

    if (allBlocks.empty()) return splitBlocks;

    vector<HyperBlock> currentGroup;
    int currentClass = allBlocks[0].classNum;

    for (const auto& block : allBlocks) {
        if (block.classNum != currentClass) {
            // Class label changed. finalize current group
            splitBlocks.push_back(currentGroup);
            currentGroup.clear();
            currentClass = block.classNum;
        }
        currentGroup.push_back(block);
    }

    // Push the last group
    if (!currentGroup.empty()) {
        splitBlocks.push_back(currentGroup);
    }

    return splitBlocks;
}

/***
* We want to go through the hyperBlocks that were generated and write them to a file.
*
*
* This print isn't caring about disjunctive blocks.
*/
void DataUtil::saveBasicHBsToCSV(const vector<HyperBlock>& hyperBlocks, const string& fileName, int FIELD_LENGTH){
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


void DataUtil::saveOneToOneHBsToCSV(const vector<vector<HyperBlock>>& oneToOneHBs, const string& fileName, int FIELD_LENGTH){
    ofstream file(fileName);
    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return;
    }

    for (const auto& blockSet : oneToOneHBs) {
        if (blockSet.empty()) continue;

        // --- FIXED HEADER SAVING --- //
        set<int> classNums;
        for (const auto& hb : blockSet) {
            classNums.insert(hb.classNum);
            if (classNums.size() == 2) break;
        }

        if (classNums.size() != 2) {
            cerr << "Warning: Expected 2 classes in block set but found " << classNums.size() << endl;
        }

        auto it = classNums.begin();
        file << "# " << *it << "," << *(++it) << "\n";
        // --------------------------- //

        for (const auto& hyperBlock : blockSet) {
            for (const vector<float>& min : hyperBlock.minimums) {
                file << min[0] << ",";
            }

            for (const vector<float>& max : hyperBlock.maximums) {
                file << max[0] << ",";
            }

            file << hyperBlock.classNum << "\n";
        }

        file << "\n"; // Separate block sets
    }

    file.close();
}

/**
 * Binary format for saving and loading One-to-One HyperBlock sets.
 *
 * PURPOSE:
 * This format is used to serialize multiple sets of binary classification HyperBlocks,
 * where each set corresponds to a unique class-pair (e.g., class 0 vs class 1).
 * Each HyperBlock consists of a set of minimum and maximum bounds per attribute,
 * and a target class label indicating which class it covers.
 *
 * FILE STRUCTURE (all integers are stored as 4-byte `int`, floats as 4-byte `float`):
 *
 * [int]   numBlockSets                      // Total number of class-pair HyperBlock sets
 *
 * For each block set:
 *     [int] classA                          // First class in the binary pair
 *     [int] classB                          // Second class in the binary pair
 *     [int] numBlocks                       // Number of HyperBlocks in this class-pair set
 *
 *     For each HyperBlock:
 *         [int] attrCount                  // Number of attributes in the block
 *         [float] min_0                    // First attribute minimum bound
 *         ...
 *         [float] min_(attrCount-1)
 *         [float] max_0                    // First attribute maximum bound
 *         ...
 *         [float] max_(attrCount-1)
 *         [int] classNum                   // Class this block belongs to (same as classA or classB)
 *
 * NOTES:
 * - Each `minimums` and `maximums` vector is assumed to contain one float per attribute.
 * - The file format does NOT currently support disjunctive/merged bounds per attribute.
 * - HyperBlock is assumed to be constructible from (maximums, minimums, classNum).
 *
 */
void DataUtil::saveOneToOneHBsToBinary(const vector<vector<HyperBlock>>& oneToOneHBs, const string& fileName) {
    ofstream out(fileName, ios::binary);
    if (!out.is_open()) {
        cerr << "Error opening binary file for write: " << fileName << endl;
        return;
    }

    int numBlockSets = static_cast<int>(oneToOneHBs.size());
    out.write(reinterpret_cast<char*>(&numBlockSets), sizeof(int));

    for (const auto& blockSet : oneToOneHBs) {
        // Save class pair
        set<int> classNums;
        for (const auto& hb : blockSet) {
            classNums.insert(hb.classNum);
            if (classNums.size() == 2) break;
        }

        int classA = *classNums.begin();
        int classB = *(++classNums.begin());
        out.write(reinterpret_cast<char*>(&classA), sizeof(int));
        out.write(reinterpret_cast<char*>(&classB), sizeof(int));

        int numBlocks = static_cast<int>(blockSet.size());
        out.write(reinterpret_cast<char*>(&numBlocks), sizeof(int));

        for (const auto& hb : blockSet) {
            int attrCount = static_cast<int>(hb.minimums.size());
            out.write(reinterpret_cast<char*>(&attrCount), sizeof(int));

            // Write mins
            for (const auto& vec : hb.minimums) {
                float val = vec[0];
                out.write(reinterpret_cast<char*>(&val), sizeof(float));
            }

            // Write maxs
            for (const auto& vec : hb.maximums) {
                float val = vec[0];
                out.write(reinterpret_cast<char*>(&val), sizeof(float));
            }

            out.write(reinterpret_cast<const char*>(&hb.classNum), sizeof(int));
        }
    }

    out.close();
}


/**
* This will load the One-To-One HyperBlocks from their file.
*/
vector<vector<HyperBlock>> DataUtil::loadOneToOneHBsFromCSV(const string& fileName,vector<pair<int, int>>& classPairsOut) {
    ifstream file(fileName);
    vector<vector<HyperBlock>> allHyperBlocks;
    vector<HyperBlock> currentSet;

    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return allHyperBlocks;
    }

    string line;
    int currentClassA = -1;
    int currentClassB = -1;

    while (getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (line.empty()) {
            // End of current set
            if (!currentSet.empty()) {
                allHyperBlocks.push_back(currentSet);
                classPairsOut.emplace_back(currentClassA, currentClassB);
                currentSet.clear();
            }
            continue;
        }

        if (line[0] == '#') {
            // Metadata line, extract classes
            stringstream ss(line.substr(1)); // Removes the # tag
            string token;
            if (getline(ss, token, ',')) {
                currentClassA = stoi(token);
            }
            if (getline(ss, token, ',')) {
                currentClassB = stoi(token);
            }
            continue;
        }

        // Normal block line
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

        int num_attributes = (temp_vals.size() - 1) / 2;
        int classNum = static_cast<int>(temp_vals.back());
        temp_vals.pop_back(); // Remove classNum

        for (int i = 0; i < num_attributes; ++i) {
            minimums.push_back({ temp_vals[i] });
            maximums.push_back({ temp_vals[i + num_attributes] });
        }

        // THIS MAKES THE HYPERBLOCKS - emplace_back calls the constructor and makes it in place in the set.
        currentSet.emplace_back(maximums, minimums, classNum);
    }

    // Push last set if the file didn't end with a blank line
    if (!currentSet.empty()) {
        allHyperBlocks.push_back(currentSet);
        classPairsOut.emplace_back(currentClassA, currentClassB);
    }

    file.close();
    return allHyperBlocks;
}


vector<vector<HyperBlock>> DataUtil::loadOneToOneHBsFromBinary(const string& fileName, vector<pair<int, int>>& classPairsOut) {
    ifstream in(fileName, ios::binary);
    vector<vector<HyperBlock>> allHyperBlocks;

    if (!in.is_open()) {
        cerr << "Error opening binary file for read: " << fileName << endl;
        return allHyperBlocks;
    }

    int numBlockSets;
    in.read(reinterpret_cast<char*>(&numBlockSets), sizeof(int));

    for (int setIdx = 0; setIdx < numBlockSets; ++setIdx) {
        int classA, classB;
        in.read(reinterpret_cast<char*>(&classA), sizeof(int));
        in.read(reinterpret_cast<char*>(&classB), sizeof(int));
        classPairsOut.emplace_back(classA, classB);

        int numBlocks;
        in.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));

        vector<HyperBlock> currentSet;

        for (int b = 0; b < numBlocks; ++b) {
            int attrCount;
            in.read(reinterpret_cast<char*>(&attrCount), sizeof(int));

            vector<vector<float>> mins(attrCount);
            vector<vector<float>> maxs(attrCount);

            for (int i = 0; i < attrCount; ++i) {
                float val;
                in.read(reinterpret_cast<char*>(&val), sizeof(float));
                mins[i].push_back(val);
            }

            for (int i = 0; i < attrCount; ++i) {
                float val;
                in.read(reinterpret_cast<char*>(&val), sizeof(float));
                maxs[i].push_back(val);
            }

            int classNum;
            in.read(reinterpret_cast<char*>(&classNum), sizeof(int));

            currentSet.emplace_back(maxs, mins, classNum);
        }

        allHyperBlocks.push_back(currentSet);
    }

    in.close();
    return allHyperBlocks;
}


/**
* Find the min/max values in each column of data across the dataset.
* Can use this in normalization and also for making sure test set is normalized with
* the same values as the training set.
*/
void DataUtil::findMinMaxValuesInDataset(const vector<vector<vector<float>>>& dataset, vector<float>& minValues, vector<float>& maxValues, int FIELD_LENGTH) {
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


// just used to mark if a column has all same value
vector<bool> DataUtil::markUniformColumns(const vector<vector<vector<float>>>& data) {
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

    return removed;
}

// special flattening function. This one flattens the min and max vounds of an HB. but in a goofy way.
// since we have to account for disjunctions, we always have a count of how many of each attribute are in each interval.
// for example a regular Hb which has no disjunction would still look like mins[1.0, 0.2, 1.0, 0.4, 1.0, 0.65, 1.0, 0.2].
// all the 1.0's just show that each attribute had one value for min. in a disjunction it would be [2.0, 0.4, 0.75, 1.0, 0.6] and so on.
// also has a bit of a crazy return. we return the flattened mins and maxes of our list of HBs, but then also two more arrays.one for the block edges.
// block edges array tells us where each HB starts and ends. for example block 0 is at index 0 obviously. But then block 1 may be at 4, and then block 2 could be at index 9 if we had a disjunction somewhere.
// block classes is just the int value of each one, casted as a float. each block's class is in this vector.
vector<vector<float>> DataUtil::flattenMinsMaxesForRUB(vector<HyperBlock>& hyper_blocks, int FIELD_LENGTH){
    // Declare vectors
    vector<float> flatMinsList;
    vector<float> flatMaxesList;
    vector<float> blockEdges;
    vector<float> blockClasses(hyper_blocks.size());
    vector<float> intervalCounts;

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
            intervalCounts.push_back(numIntervals);
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
    result.push_back(move(intervalCounts));
    return result;
}

// just flattens all our data into a vector<vector<float>>. we have to keep a tracker of where each class started and ended.
// that is the point of the class border. so say class 1's start will be index 600 of the flattened set, classBorder[1] == 600.
vector<vector<float>> DataUtil::flattenDataset(vector<vector<vector<float>>>& data) {
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


// our function to flatten our list of HBs without encoding lengths in. this is what we use for removing attirbutes
// this flattens the HBs very nicely. it doesn't encode the count of each attribute, because we use it BEFORE we could ever have a disjunction.
vector<vector<float>> DataUtil::flatMinMaxNoEncode(vector<HyperBlock> hyper_blocks, int FIELD_LENGTH) {

    int size = hyper_blocks.size();
    vector<float> flatMinsList;
    vector<float> flatMaxesList;
    vector<float> blockClasses(size, 0.0f);

    const int numAttributes = hyper_blocks.at(0).minimums.size();

    // Process each hyper block
    for (size_t hb = 0; hb < size; hb++) {
        HyperBlock& block = hyper_blocks[hb];
        // cast the classNum as a float so that we can put it in the vector of floats we are returning
        blockClasses[hb] = static_cast<float>(block.classNum);

        // Iterate through each attribute in the block
        for (size_t m = 0; m < numAttributes; m++) {
            // Record the count of intervals for the current attribute
            flatMinsList.push_back(block.minimums[m][0]);
            flatMaxesList.push_back(block.maximums[m][0]);
        }
    }

    // Return the five arrays in a vector (order matches the original Java return)
    return { flatMinsList, flatMaxesList,  blockClasses };
}




void DataUtil::splitTrainTestByPercent(vector<vector<vector<float>>>& trainingData, vector<vector<vector<float>>>& testingData, float percentTrain) {
    mt19937 rng(42);
    testingData.clear();
    testingData.resize(trainingData.size());

    for (size_t classIdx = 0; classIdx < trainingData.size(); classIdx++) {
        auto& classPoints = trainingData[classIdx];

        // Shuffle points for randomness
        shuffle(classPoints.begin(), classPoints.end(), rng);

        int originalSize = classPoints.size();
        int newTrainCount = static_cast<int>(originalSize * percentTrain);

        // Move points from trainingData to testingData
        while (classPoints.size() > newTrainCount) {
            testingData[classIdx].push_back(move(classPoints.back()));
            classPoints.pop_back();
        }
    }
}


// Splits an already-normalized dataset into k folds with stratified sampling.
// The input 'dataset' is expected to be organized as: [class][point][attribute].
// The returned 4D vector is structured as: [fold][class][point][attribute].
vector<vector<vector<vector<float>>>> DataUtil::splitDataset(const vector<vector<vector<float>>> &dataset, int k) {
    // Create a 4D vector with k folds.
    // Each fold contains one vector per class.
    vector<vector<vector<vector<float>>>> folds(k, vector<vector<vector<float>>>(dataset.size()));

    // Use a fixed random seed for reproducibility.
    mt19937 rng(42);

    // For each class, shuffle its points and distribute them round-robin into the k folds.
    for (size_t classIdx = 0; classIdx < dataset.size(); classIdx++) {
        // Copy the points for this class.
        vector<vector<float>> points = dataset[classIdx];

        // Shuffle the points.
        shuffle(points.begin(), points.end(), rng);

        // Distribute the points into folds.
        for (int i = 0; i < points.size(); i++) {
            int foldIndex = i % k;
            folds[foldIndex][classIdx].push_back(points[i]);
        }
    }

    return folds;
}

/**
 * This is a Condensed Nearest Neighbors implementation CNN. Essentially it will
 * allow us to reduce the number of points in a dataset like MNIST by building a representative set.
 *
 * It is being tested because it preserves decision boundaries for 1-Nearest Neighbors, and might be good for HB
 * generation speed ups. This also comes with a risk though of overfitting to the training data, so if used in the future please
 * read more about CNN.
 *
 * The following is a solid article that could help:
 * https://abhic159.medium.com/cnn-condensed-nearest-neighbors-3261bd0c39fb
 *
 * @param dataset The training dataset for the HBs.
 * @param FIELD_LENGTH The number of attributes in the dataset.
 * @return The reduced size training dataset.
 */
std::vector<std::vector<std::vector<float>>> DataUtil::condensedNearestNeighbors(
    const std::vector<std::vector<std::vector<float>>>& dataset,
    int FIELD_LENGTH
) {
    typedef std::vector<float> Point;
    typedef std::pair<Point, int> LabeledPoint;

    std::vector<LabeledPoint> condensedSet;

    // Step 1: Add one sample per class
    for (int cls = 0; cls < static_cast<int>(dataset.size()); ++cls) {
        if (!dataset[cls].empty()) {
            condensedSet.push_back(std::make_pair(dataset[cls][0], cls));
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (int cls = 0; cls < static_cast<int>(dataset.size()); ++cls) {
            for (int p = 0; p < static_cast<int>(dataset[cls].size()); ++p) {
                const Point& pt = dataset[cls][p];

                // Skip if this point is already in the condensed set
                bool alreadyIn = false;
                for (int i = 0; i < static_cast<int>(condensedSet.size()); ++i) {
                    if (pt == condensedSet[i].first && cls == condensedSet[i].second) {
                        alreadyIn = true;
                        break;
                    }
                }
                if (alreadyIn) continue;

                // 1-NN classification
                float bestDist = std::numeric_limits<float>::max();
                int bestLabel = -1;

                for (int i = 0; i < static_cast<int>(condensedSet.size()); ++i) {
                    const Point& ref = condensedSet[i].first;
                    int refLabel = condensedSet[i].second;

                    float dist = 0.0f;
                    for (int d = 0; d < FIELD_LENGTH; ++d) {
                        float diff = pt[d] - ref[d];
                        dist += diff * diff;
                    }

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestLabel = refLabel;
                    }
                }

                // If misclassified, add to condensed set
                if (bestLabel != cls) {
                    condensedSet.push_back(std::make_pair(pt, cls));
                    changed = true;
                }
            }
        }
    }

    // Reconstruct the 3D vector [class][point][attr]
    std::vector<std::vector<std::vector<float>>> result(dataset.size());
    for (int i = 0; i < static_cast<int>(condensedSet.size()); ++i) {
        const Point& pt = condensedSet[i].first;
        int cls = condensedSet[i].second;
        result[cls].push_back(pt);
    }

    return result;
}