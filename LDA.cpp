#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include <future>
using namespace std;

// Matrix inverse using naive Gauss-Jordan elimination
vector<vector<float>> inverse(const vector<vector<float>>& matrix) {
    int n = matrix.size();
    vector<vector<float>> augmented(n, vector<float>(2 * n, 0.0f));

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0f;  // identity portion
    }

    // Forward elimination
    for (int i = 0; i < n; i++) {
        float pivot = augmented[i][i];
        if (fabs(pivot) < 1e-12) {
            cerr << "Matrix is singular or near-singular.\n";
            exit(EXIT_FAILURE);
        }
        // Normalize pivot row
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        // Eliminate down
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract inverse
    vector<vector<float>> inv(n, vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i][j] = augmented[i][j + n];
        }
    }
    return inv;
}

// Simple matrix-vector multiply
vector<float> matrixVectorMultiply(const vector<vector<float>>& A, const vector<float>& v) {
    int rows = A.size();
    int cols = A[0].size();
    vector<float> result(rows, 0.0f);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

// Helper: dot product of two vectors
inline float dotProduct(const vector<float>& a, const vector<float>& b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// --------------------------------------
// 2. Core two-class LDA function (with bias computation)
// --------------------------------------

// Define a struct to hold both the weight vector and its bias.
struct LDAClassifier {
    vector<float> w; // the coefficients of the classifier
    float bias; // offsets the threshold back to 0 so that we are comparing fairly
};

/**
 * Computes a single LDA vector w (and bias) that separates two classes:
 *   classA vs. classB
 * Returns an LDAClassifier containing the weight vector and bias.
 */
LDAClassifier computeBinaryLDA(const vector<vector<float>>& classA,
                               const vector<vector<float>>& classB)
{
    int numFeatures = classA[0].size();

    // 1. Compute mean vectors of class A and class B
    vector<float> meanA(numFeatures, 0.0f);
    for (auto& row : classA) {
        for (int j = 0; j < numFeatures; j++) {
            meanA[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        meanA[j] /= (float)classA.size();
    }

    vector<float> meanB(numFeatures, 0.0f);
    for (auto& row : classB) {
        for (int j = 0; j < numFeatures; j++) {
            meanB[j] += row[j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        meanB[j] /= (float)classB.size();
    }

    // 2. Compute within-class scatter Sw = S_A + S_B
    vector<vector<float>> Sw(numFeatures, vector<float>(numFeatures, 0.0f));
    auto addScatter = [&](const vector<vector<float>>& data, const vector<float>& mean) {
        for (auto& x : data) {
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    Sw[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]);
                }
            }
        }
    };

    addScatter(classA, meanA);
    addScatter(classB, meanB);

    // REGULARIZE TO AVOID THE MATRIX BEING SINGULAR.
    float lambda = 1e-3f; // adjust as needed.
    for (int i = 0; i < numFeatures; i++) {
        Sw[i][i] += lambda;
    }

    // 3. Compute w = Sw^-1 * (meanB - meanA)
    vector<vector<float>> SwInv = inverse(Sw);
    vector<float> meanDiff(numFeatures, 0.0f);
    for (int j = 0; j < numFeatures; j++) {
        meanDiff[j] = meanB[j] - meanA[j];
    }
    vector<float> w = matrixVectorMultiply(SwInv, meanDiff);

    // 4. Normalize w
    float normVal = 0.0f;
    for (float val : w) {
        normVal += val * val;
    }
    normVal = sqrt(normVal);
    if (normVal > 1e-12) {
        for (float &val : w) {
            val /= normVal;
        }
    }

    // 5. Compute bias: use the midpoint between the projections of the class means
    float projMeanA = dotProduct(w, meanA);
    float projMeanB = dotProduct(w, meanB);
    float bias = (projMeanA + projMeanB) / 2.0f;

    // Flip w and bias if the target class's projection is lower than the other
    if (projMeanA < projMeanB) {
        for (float &val : w) {
            val = -val;
        }
        bias = -bias;
    }

    LDAClassifier classifier;
    classifier.w = w;
    classifier.bias = bias;
    return classifier;
}

// predictClass: for a given sample, compute the adjusted score for each classifier
// (w^T x - bias) and return the class with the highest score.
int predictClass(const vector<LDAClassifier>& classifiers, const vector<float>& sample) {
    int bestClass = -1;
    float bestScore = -1e9f; // a very low starting score
    for (size_t i = 0; i < classifiers.size(); i++) {
        float score = dotProduct(classifiers[i].w, sample) - classifiers[i].bias;
        if (score > bestScore) {
            bestScore = score;
            bestClass = static_cast<int>(i);
        }
    }
    return bestClass;
}

// testMultiClassAccuracy: loops over each sample in inputData (labeled by class),
// predicts its class using predictClass, and computes overall accuracy.
void testMultiClassAccuracy(const vector<LDAClassifier>& classifiers,
                            const vector<vector<vector<float>>>& inputData) {
    int numClasses = inputData.size();
    int totalSamples = 0;
    int correct = 0;

    for (int i = 0; i < numClasses; i++) {
        for (const auto &sample : inputData[i]) {
            int predicted = predictClass(classifiers, sample);
            if (predicted == i) {
                correct++;
            }
            totalSamples++;
        }
    }

    float accuracy = static_cast<float>(correct) / totalSamples;
    cout << "Overall multi-class accuracy: " << accuracy * 100.0f << "%" << endl;
}


// --------------------------------------
// “One-vs-Rest” LDA for multi-class
// --------------------------------------
/**
 * Given inputData where inputData[i] = all samples of class i,
 * run a 2-class LDA for each class i vs. all other classes combined.
 *
 * Returns a vector of LDAClassifier, one per class.
*/
vector<vector<float>> linearDiscriminantAnalysis(const vector<vector<vector<float>>>& inputData) {
    int numClasses = inputData.size();
    vector<future<LDAClassifier>> futures;
    futures.reserve(numClasses);

    // Launch each binary classifier computation asynchronously.
    for (int i = 0; i < numClasses; i++) {
        futures.push_back(std::async(std::launch::async, [&, i]() -> LDAClassifier {
            // classA = samples of class i.
            const auto& classA = inputData[i];
            // Build classB as the union of all samples from other classes.
            vector<vector<float>> classB;
            for (int j = 0; j < numClasses; j++) {
                if (j == i) continue;
                classB.insert(classB.end(), inputData[j].begin(), inputData[j].end());
            }
            // Compute and return the binary LDA classifier for class i vs. rest.
            return computeBinaryLDA(classA, classB);
        }));
    }

    // Retrieve the results from each future.
    vector<LDAClassifier> classifiers;
    classifiers.reserve(numClasses);
    for (int i = 0; i < numClasses; i++) {
        classifiers.push_back(futures[i].get());
    }

    // Extract separation vectors (weight vectors) from the classifiers.
    vector<vector<float>> separationVectors;
    for (const LDAClassifier &classifier : classifiers) {
        separationVectors.push_back(classifier.w);
    }

    testMultiClassAccuracy(classifiers, inputData);
    return separationVectors;
}

