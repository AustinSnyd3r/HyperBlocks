//
// Created by asnyd on 3/20/2025.
//
#include "DataUtil.h"

extern int FIELD_LENGTH;
extern int NUM_CLASSES;




// In-place stratified split: modifies trainingData by moving points to validationData
void DataUtil::createValidationSplit(std::vector<std::vector<std::vector<float>>>& trainingData, std::vector<std::vector<std::vector<float>>>& validationData, float validationFraction, unsigned int randomSeed) {
    std::mt19937 rng(randomSeed);
    validationData.clear();
    validationData.resize(trainingData.size());

    // Go through each class
    for (int classIdx = 0; classIdx < trainingData.size(); ++classIdx) {
        auto& pointsInClass = trainingData[classIdx];

        std::vector<int> indices(pointsInClass.size());
        for (int i = 0; i < pointsInClass.size(); ++i)
            indices[i] = i;

        std::shuffle(indices.begin(), indices.end(), rng);

        int valCount = static_cast<int>(pointsInClass.size() * validationFraction);

        // Move validation points
        for (int i = 0; i < valCount; ++i) {
            validationData[classIdx].push_back(std::move(pointsInClass[indices[i]]));
        }

        // Erase validation points from training set
        // Important: sort indices in reverse so erase does not invalidate remaining indices
        std::sort(indices.begin(), indices.begin() + valCount, std::greater<int>());
        for (int i = 0; i < valCount; ++i) {
            pointsInClass.erase(pointsInClass.begin() + indices[i]);
        }
    }
}



/*  Returns a class seperated version of the dataset
 *  Each class has an entry in the outer vector with a 2-d vector of its points
 */
std::vector<std::vector<std::vector<float>>> DataUtil::dataSetup(const std::string filepath, std::map<std::string, int>& classMap, std::map<int, std::string>& reversedClassMap) {
    // 3D std::std::vector: data[class][point][attribute]
    std::vector<std::vector<std::vector<float>>> data;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filepath << std::endl;
        return data;
    }

    int classNum = 0;
    std::string line;
    // Ignore the header, can use later if needed
    getline(file, line);

    // Read through all rows of CSV
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        // Read the entire row, splitting by commas
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Skip empty lines
        if (row.empty()) continue;

        std::string classLabel = row.back();
        row.pop_back();

        // Check if class exists, else create new entry
        if (classMap.count(classLabel) == 0) {
            classMap[classLabel] = classNum;
            data.push_back(std::vector<std::vector<float>>());
            classNum++;
        }

        int classIndex = classMap[classLabel];

        std::vector<float> point;
        for (const std::string& val : row) {
            try {
                point.push_back(stof(val));  // Convert to float and add to the point
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid value '" << val << "' in CSV" << std::endl;
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
void DataUtil::saveHyperBlocksToFile(const std::string& filepath, const std::vector<std::vector<std::vector<float>>>& hyperBlocks) {
    std::ofstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
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
    std::cout << "Hyperblocks saved to " << filepath << std::endl;
}

/**
* This will save the normalized dataset back so that we can use the same one in DV with the same normalization.
*/
void DataUtil::saveNormalizedVersionToCsv(std::string fileName, std::vector<std::vector<std::vector<float>>>& data) {
    std::ofstream outFile(fileName);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
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

std::vector<HyperBlock> DataUtil::loadBasicHBsFromCSV(const std::string& fileName) {
    std::ifstream file(fileName);
    std::vector<HyperBlock> hyperBlocks;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return hyperBlocks;
    }

    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::vector<float>> minimums, maximums;
        std::string value;
        std::vector<float> temp_vals;

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
void DataUtil::normalizeTestSet(std::vector<std::vector<std::vector<float>>>& testSet, const std::vector<float>& minValues, const std::vector<float>& maxValues, int FIELD_LENGTH) {
    if (testSet.empty()){
      std::cout << "Test set was empty when trying to normalize" << std::endl;
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
	         std::cout << "Out of range" << point[k] << std::endl;
             point[k] = 1.000000f;
		  }
          if(point[k] < 0.0f){
		  	point[k] = 0.000000f;
          }
        }
      }
    }




}


void DataUtil::minMaxNormalization(std::vector<std::vector<std::vector<float>>>& dataset, const std::vector<float>& minValues, const std::vector<float>& maxValues, int FIELD_LENGTH) {
    std::cout << "Normalizing the dataset" << std::endl;
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


// Function to reorder testing dataset based on training class std::mapping
std::vector<std::vector<std::vector<float>>> DataUtil::reorderTestingDataset(const std::vector<std::vector<std::vector<float>>>& testingData, const std::map<std::string, int>& CLASS_MAP_TRAINING, const std::map<std::string, int>& CLASS_MAP_TESTING) {
    // Create a new std::vector with the same size as the testing data
    std::vector<std::vector<std::vector<float>>> reorderedTestingData(testingData.size());

    // Create a mapping from testing indices to training indices
    std::map<int, int> indexMap;
    // For each (className -> trainingIndex) in the training map
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


std::vector<HyperBlock> DataUtil::loadBasicHBsFromBinary(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::binary);
    std::vector<HyperBlock> loadedBlocks;

    if (!file.is_open()) {
        std::cerr << "Error opening binary file: " << fileName << std::endl;
        return loadedBlocks;
    }

    int numBlocks, fieldLength;
    file.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));
    file.read(reinterpret_cast<char*>(&fieldLength), sizeof(int));

    for (int b = 0; b < numBlocks; ++b) {
        std::vector<std::vector<float>> mins(fieldLength, std::vector<float>(1));
        std::vector<std::vector<float>> maxs(fieldLength, std::vector<float>(1));

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

void DataUtil::saveBasicHBsToBinary(const std::vector<HyperBlock>& hyperBlocks, const std::string& fileName, int FIELD_LENGTH) {
    std::ofstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening binary file: " << fileName << std::endl;
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

std::vector<std::vector<HyperBlock>> DataUtil::loadOneToSomeBlocksFromBinary(const std::string& fileName) {
    std::vector<HyperBlock> allBlocks = DataUtil::loadBasicHBsFromCSV(fileName);
    std::vector<std::vector<HyperBlock>> splitBlocks;

    if (allBlocks.empty()) return splitBlocks;

    std::vector<HyperBlock> currentGroup;
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
void DataUtil::saveBasicHBsToCSV(const std::vector<HyperBlock>& hyperBlocks, const std::string& fileName, int FIELD_LENGTH){
	// Open file for writing
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }

	// min1, min2, min3, ..., minN, max1, max2, max3, ..., maxN, class
	for (const auto& hyperBlock : hyperBlocks) {
        // Write minimums
        for (const std::vector<float>& min : hyperBlock.minimums) {
            file << min[0] << ",";
        }

        // Write maximums
        for (const std::vector<float>& max : hyperBlock.maximums) {
            file << max[0] << ",";
        }

        // Write the class number
        file << hyperBlock.classNum << "\n";
    }

    file.close();
}


void DataUtil::saveOneToOneHBsToCSV(const std::vector<std::vector<HyperBlock>>& oneToOneHBs, const std::string& fileName, int FIELD_LENGTH){
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }

    for (const auto& blockSet : oneToOneHBs) {
        if (blockSet.empty()) continue;

        // --- FIXED HEADER SAVING --- //
        std::set<int> classNums;
        for (const auto& hb : blockSet) {
            classNums.insert(hb.classNum);
            if (classNums.size() == 2) break;
        }

        if (classNums.size() != 2) {
            std::cerr << "Warning: Expected 2 classes in block set but found " << classNums.size() << std::endl;
        }

        auto it = classNums.begin();
        file << "# " << *it << "," << *(++it) << "\n";
        // --------------------------- //

        for (const auto& hyperBlock : blockSet) {
            for (const std::vector<float>& min : hyperBlock.minimums) {
                file << min[0] << ",";
            }

            for (const std::vector<float>& max : hyperBlock.maximums) {
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
void DataUtil::saveOneToOneHBsToBinary(const std::vector<std::vector<HyperBlock>>& oneToOneHBs, const std::string& fileName) {
    std::ofstream out(fileName, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error opening binary file for write: " << fileName << std::endl;
        return;
    }

    int numBlockSets = static_cast<int>(oneToOneHBs.size());
    out.write(reinterpret_cast<char*>(&numBlockSets), sizeof(int));

    for (const auto& blockSet : oneToOneHBs) {
        // Save class pair
        std::set<int> classNums;
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
std::vector<std::vector<HyperBlock>> DataUtil::loadOneToOneHBsFromCSV(const std::string& fileName,std::vector<std::pair<int, int>>& classPairsOut) {
    std::ifstream file(fileName);
    std::vector<std::vector<HyperBlock>> allHyperBlocks;
    std::vector<HyperBlock> currentSet;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return allHyperBlocks;
    }

    std::string line;
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
            std::stringstream ss(line.substr(1)); // Removes the # tag
            std::string token;
            if (getline(ss, token, ',')) {
                currentClassA = stoi(token);
            }
            if (getline(ss, token, ',')) {
                currentClassB = stoi(token);
            }
            continue;
        }

        // Normal block line
        std::stringstream ss(line);
        std::vector<std::vector<float>> minimums, maximums;
        std::string value;
        std::vector<float> temp_vals;

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


std::vector<std::vector<HyperBlock>> DataUtil::loadOneToOneHBsFromBinary(const std::string& fileName, std::vector<std::pair<int, int>>& classPairsOut) {
    std::ifstream in(fileName, std::ios::binary);
    std::vector<std::vector<HyperBlock>> allHyperBlocks;

    if (!in.is_open()) {
        std::cerr << "Error opening binary file for read: " << fileName << std::endl;
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

        std::vector<HyperBlock> currentSet;

        for (int b = 0; b < numBlocks; ++b) {
            int attrCount;
            in.read(reinterpret_cast<char*>(&attrCount), sizeof(int));

            std::vector<std::vector<float>> mins(attrCount);
            std::vector<std::vector<float>> maxs(attrCount);

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
void DataUtil::findMinMaxValuesInDataset(const std::vector<std::vector<std::vector<float>>>& dataset, std::vector<float>& minValues, std::vector<float>& maxValues, int FIELD_LENGTH) {
    // Step 1: Find min and max for each attribute
    for (const auto& class_data : dataset) {
        for (const auto& point : class_data) {
            for (int k = 0; k < FIELD_LENGTH; k++) {
                minValues[k] = std::min(minValues[k], point[k]);
                maxValues[k] = std::max(maxValues[k], point[k]);
            }
        }
    }
}



std::vector<bool> DataUtil::markUniformColumns(const std::vector<std::vector<std::vector<float>>>& data) {
    // std::cout << "Starting mark uniform columns\n" << std::endl;

    if (data.empty() || data[0].empty()) return std::vector<bool>(); // Handle edge case

    int numCols = data[0][0].size();
    std::vector<bool> removed(numCols, false);

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


std::vector<std::vector<float>> DataUtil::flattenMinsMaxesForRUB(std::vector<HyperBlock>& hyper_blocks, int FIELD_LENGTH){
    // Declare std::vectors
    std::vector<float> flatMinsList;
    std::vector<float> flatMaxesList;
    std::vector<float> blockEdges;
    std::vector<float> blockClasses(hyper_blocks.size());

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
    std::vector<std::vector<float>> result;
    result.push_back(std::move(flatMinsList));
    result.push_back(std::move(flatMaxesList));
    result.push_back(std::move(blockEdges));
    result.push_back(std::move(blockClasses));
    return result;
}


std::vector<std::vector<float>> DataUtil::flattenDataset(std::vector<std::vector<std::vector<float>>>& data) {
    std::vector<float> dataset;
    std::vector<float> classBorder(data.size() + 1);
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
    std::vector<std::vector<float>> result;
    result.push_back(move(dataset));
    result.push_back(move(classBorder));
    return result;
}


// our function to flatten our list of HBs without encoding lengths in. this is what we use for removing attirbutes
std::vector<std::vector<float>> DataUtil::flatMinMaxNoEncode(std::vector<HyperBlock> hyper_blocks, int FIELD_LENGTH) {

    int size = hyper_blocks.size();
    std::vector<float> flatMinsList;
    std::vector<float> flatMaxesList;
    std::vector<float> blockEdges(size + 1, 0.0f);
    std::vector<float> blockClasses(size, 0.0f);
    std::vector<float> intervalCounts(size * FIELD_LENGTH, 0.0f);

    // First block starts at 0.
    blockEdges[0] = 0.0f;
    int idx = 0;

    // Process each hyper block
    for (size_t hb = 0; hb < size; hb++) {
        HyperBlock& block = hyper_blocks[hb];
        // cast the classNum as a float so that we can put it in the std::vector of floats we are returning
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


void DataUtil::splitTrainTestByPercent(std::vector<std::vector<std::vector<float>>>& trainingData, std::vector<std::vector<std::vector<float>>>& testingData, float percentTrain) {
    std::mt19937 rng(42);
    testingData.clear();
    testingData.resize(trainingData.size());

    for (size_t classIdx = 0; classIdx < trainingData.size(); classIdx++) {
        auto& classPoints = trainingData[classIdx];

        // Shuffle points for randomness
        std::shuffle(classPoints.begin(), classPoints.end(), rng);

        int originalSize = classPoints.size();
        int newTrainCount = static_cast<int>(originalSize * percentTrain);

        // Move points from trainingData to testingData
        while (classPoints.size() > newTrainCount) {
            testingData[classIdx].push_back(std::move(classPoints.back()));
            classPoints.pop_back();
        }
    }
}


// Splits an already-normalized dataset into k folds with stratified sampling.
// The input 'dataset' is expected to be organized as: [class][point][attribute].
// The returned 4D vector is structured as: [fold][class][point][attribute].
std::vector<std::vector<std::vector<std::vector<float>>>> DataUtil::splitDataset(const std::vector<std::vector<std::vector<float>>> &dataset, int k) {
    // Create a 4D vector with k folds.
    // Each fold contains one vector per class.
    std::vector<std::vector<std::vector<std::vector<float>>>> folds(k, std::vector<std::vector<std::vector<float>>>(dataset.size()));

    // Use a fixed random seed for reproducibility.
    std::mt19937 rng(42);

    // For each class, shuffle its points and distribute them round-robin into the k folds.
    for (size_t classIdx = 0; classIdx < dataset.size(); classIdx++) {
        // Copy the points for this class.
        std::vector<std::vector<float>> points = dataset[classIdx];

        // Shuffle the points.
        std::shuffle(points.begin(), points.end(), rng);

        // Distribute the points into folds.
        for (int i = 0; i < points.size(); i++) {
            int foldIndex = i % k;
            folds[foldIndex][classIdx].push_back(points[i]);
        }
    }

    return folds;
}