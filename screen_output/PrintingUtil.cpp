//
// Created by asnyd on 3/20/2025.
//
#include <iomanip>
#include "PrintingUtil.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <limits>

// Function to clear the console screen (cross-platform)
void PrintingUtil::clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

// Function to wait for user input before continuing
void PrintingUtil::waitForEnter() {
    std::cout << "\nPress Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// Function to display the main menu
void PrintingUtil::displayMainMenu() {
    clearScreen();
    std::cout << "=== HyperBlock Classification System ===\n\n";
    std::cout << "1. Import training data.\n";
    std::cout << "2. Import testing data.\n";
    std::cout << "3. Save normalized training data.\n";
    std::cout << std::endl;
    std::cout << "4. Import regular Hyperblocks.\n";
    std::cout << "5. Export regular Hyperblocks.\n";
    std::cout << "6. Generate Hyperblocks.\n";
    std::cout << "7. Simplify Hyperblocks.\n";
    std::cout << "8. Test Hyperblocks.\n";
    std::cout << "9. Test 1-1 HyperBlocks.\n";
    std::cout << "10. K Fold Cross Validation.\n";
    std::cout << std::endl;
    std::cout << "11. Generate 1-1 Hyperblocks.\n";
    std::cout << "12. Import 1-1 Hyperblocks.\n";
    std::cout << "13. Export 1-1 Hyperblocks.\n";
    std::cout << "14. K-Fold 1-1\n";
    std::cout << std::endl;
    std::cout << "15. Find Best Parameters\n\n";
    std::cout << "16. \n\n";
    std::cout << "17. Train-Test-Split\n\n";


}

float PrintingUtil::printConfusionMatrix(std::vector<std::vector<long>>& data, const int NUM_CLASSES, std::map<int, std::string>& CLASS_MAP_INT) {
    std::vector<std::string> classLabels(NUM_CLASSES);

    std::vector<float> accuracies(NUM_CLASSES, 0.0);

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
        maxWidth = std::max(maxWidth, name.length() + 2);
    }

    for (const auto& row : data) {
        for (const auto& cell : row) {
            std::string numStr = std::to_string(cell);
            maxWidth = std::max(maxWidth, numStr.length() + 2);
        }
    }

    // Print header row with "Actual\Predicted" in the corner
    std::cout << std::setw(maxWidth) << "Act\\Pred" << " |";
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << std::setw(maxWidth) << CLASS_MAP_INT[i] << " |";
    }
    std::cout << std::endl;

    // Print separator line
    std::cout << std::string(maxWidth, '-') << "-+";
    for (size_t i = 0; i < CLASS_MAP_INT.size(); i++) {
        std::cout << std::string(maxWidth, '-') << "-+";
    }
    std::cout << std::endl;

    // Print each row with row label
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << std::setw(maxWidth) << CLASS_MAP_INT[i] << " |";

        for (size_t j = 0; j < data[i].size(); j++) {
            std::cout << std::setw(maxWidth) << data[i][j] << " |";
        }

        std::cout << accuracies[i] << std::endl;
    }

    std::cout << "The overall accuracy is " << overallAccuracy << std::endl;
    return overallAccuracy;
}


void PrintingUtil::printDataset(const std::vector<std::vector<std::vector<float>>>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        std::cout << "Class " << i << ":" << std::endl;
        for (const auto& row : vec[i]) {
            std::cout << "  [";
            for (int j = 0; j < row.size(); j++) {
                std::cout << row[j];
                if (j < row.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;  // Add spacing between classes
    }
}