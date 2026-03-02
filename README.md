## Table of Contents

- [What Are Hyperblocks?](#what-are-hyperblocks)
- [Prerequisites](#prerequisites)
- [Build Instructions](#build-instructions)
- [Run Instructions](#run-instructions)
- [Command Line Compiling Instructions](#command-line-compiling-instructions)
- [Dataset Format](#dataset-format)
- [Program Usage](#program-usage)
- [Project Structure](#project-structure)
- [Contact Information & Credits](#contact-information--credits)


# Hyperblocks (HBs)

This repository, is a standalone C++/CUDA implementation of the DV2.0 Hyperblocks model, originally developed in Java at the CWU VKD Lab. It is designed to be high-performing, explainable, and cross-platform, with GPU acceleration and parallelism support.

---
##  What are HBs?

- Hyperblocks are an interpretable, rule-based machine learning model. Each hyperblock defines axis-aligned bounds (min/max) for each attribute in the dataset, forming a hyper-rectangle in feature space. 
- The following is an example, classifying the Setosa class from the Fischer Iris dataset, using only a single clause (x3).

![HyperBlock Example](images/setosa_hb_example.png)

This structure supports:
- Transparent decision-making
- Easy visualization (e.g., Parallel Coordinates)
- Rule simplification and fusion
- Compatibility with Subject-Matter Expert (SME) analysis

---
## Prerequisites

To build this project, you need:

- CMake 3.18 or higher
- CUDA Toolkit (tested with 12.6)
- A C++17-compatible compiler (GCC, Clang, MSVC)
- OpenMP (optional but recommended)

To run this project, you need:

- CUDA compatible GPU
---
##  Build Instructions

Clone the repository and run the following:

```bash
# Step 1: Create a build directory
mkdir -p build
cd build

# Step 2: Generate the build files. (may need -DCMAKE_BUILD_TYPE=Debug on Linux)
cmake ..  


# Step 3: Compile the project
cmake --build . --config Debug
```
---
##  Run Instructions

After building with CMake (inside the `build/` directory), you can run the program as follows:


Run the executable from the project root:
```bash
cd ..
Hyperblocks.exe
```
---
##  Command Line Compiling Instructions (Optional)

This section is for users who prefer to compile the program manually instead of using CMake.

### Windows (MSVC)

- **Compile**:
```bash
nvcc -Xcompiler /openmp -o a.exe ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./classification_testing/ClassificationTests.cpp -g -G -O3
```

- **Run**:
```bash
a.exe
```

### Linux

- **Compile**:
```bash
nvcc -Xcompiler -fopenmp -o a ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./classification_testing/ClassificationTests.cpp -g -G -O3 --relocatable-device-code=true
```

- **Run**:
```bash
./a
```


---


## Dataset Formatting

This section is split into:
- Dataset importing / exporting
- Hyperblock importing / exporting

### Training and Testing Datasets

Datasets used for training, testing, or classification must be in **CSV format**. The system expects:

- Each row corresponds to one data sample (point).
- Each column up to the last represents **normalized float features** in the range [0, 1].
- The **last column** must be the **class label**.

> ⚠️ If your dataset does **not** have a header row, **you must manually remove the first line**. The parser currently does **not differentiate** and will treat the first row as a header, silently discarding it.



Datasets should be placed in the `datasets/` directory. You can load them via command-line or code using utilities like `DataUtil::importData`.

---

### Hyperblock Save Files

HBs can be exported and imported in two formats:

#### 1. **Binary Format (.bin)**

- Uses `DataUtil::saveBasicHBsToBinary(...)` and `DataUtil::loadBasicHBsFromBinary(...)`
- Preserves full floating-point precision
- Best for experiments, training reuse, and deployment
- Format: 
  [int num_blocks][int num_attributes]
  [float min1, ..., minN]
  [float max1, ..., maxN]
  [int classNum]
  ... repeated for each block


#### 2. **CSV Format (.csv)**

- Uses `DataUtil::saveBasicHBsToCSV(...)` and `DataUtil::loadBasicHBsFromCSV(...)`
- Human-readable but **not precision-safe**
- When reloaded, can lead to dropped coverage due to floating point rounding
- Format (one row per block):
  min1,...,minN,max1,...,maxN,class


#### ⚠️ Important Notes:
- The loader assumes that the saved blocks match the **dimensionality** of your current dataset. No consistency check is enforced in code.
- If the HB save file and dataset do not align in number of attributes, the program **may silently fail or misclassify**.
- CSV format **should only be used for demos or visual inspection**, not for preserving exact decision boundaries.

---

### Summary Table

| Format | Precision | Human-Readable | Recommended Use |
|--------|-----------|----------------|-----------------|
| `.bin` | Full      | No             | All serious use |
| `.csv` | Lossy     | Yes            | Debug / demos   |

---

### Utility Functions Used

| Purpose         | Function Name                            |
|-----------------|-------------------------------------------|
| Import data     | `DataUtil::importData(...)`               |
| Save HBs (CSV)  | `DataUtil::saveBasicHBsToCSV(...)`        |
| Save HBs (Bin)  | `DataUtil::saveBasicHBsToBinary(...)`     |
| Load HBs (CSV)  | `DataUtil::loadBasicHBsFromCSV(...)`      |
| Load HBs (Bin)  | `DataUtil::loadBasicHBsFromBinary(...)`   |



## Program Usage
### Getting Started (Interactive Mode)

If no arguments are passed (argc < 2), the program launches into an interactive mode with a menu-driven interface. You can import datasets, normalize data, generate or simplify HBs, test on new data, or export results.

Launch the program with:
./Hyperblocks

You will see a numbered menu with options like:

- Import training/testing datasets
- Choose normalization (min-max, fixed max, or none)
- Generate new HBs
- Run test accuracy on a dataset
- Export/load precomputed blocks
- Perform K-Fold cross validation
- Run precision-weighted or 1-vs-1 classifiers

Note: The main program loops are in Host.cu. 

---

### Basic Workflow

1. Import a training dataset
  - Choose from available files in the datasets/ folder
  - Select a normalization method (min-max or fixed-scale)

2. Import a testing dataset
  - This can be normalized using training bounds or left raw
  - It is automatically aligned to correct class labels mapping via DataUtil::reorderTestingDataset(...)

3. Generate or load HBs
  - Case 6 generates traditional HBs
  - Case 4 loads from a .bin file
  - Case 11 generates 1-vs-1 HBs
  - Case 15 generates 1-vs-rest HBs

4. Simplify and save results
  - Case 7 runs simplification methods
  - Case 5 and 13 save .bin files of generated blocks

5. Run evaluation
  - Case 8 runs a test on the test set
  - Case 10 and 14 run cross-validation
  - Case 16–19 run precision-weighted evaluation or level-N experiments

---

### Running on CWU Lab Machines (SAMU140)

The lab computers in SAMU140 are equipped with NVIDIA RTX 4090 GPUs and fast CPUs, which we used for large datasets. (e.g., full MNIST runs).

Recommended setup:

1. Pre-compile the program on your own machine.
2. Load the following onto a flash drive:
  - The Hyperblocks.exe executable
  - Any datasets you want to run (e.g., MNIST .csv training/test sets)
3. On the lab machine:
  - Drag Hyperblocks.exe onto the desktop
  - Open a terminal or PowerShell window
  - Run the program directly:  
    ./Hyperblocks.exe
  - Or specify a class argument (for async mode):  
    ./Hyperblocks.exe 0

This will run the system using class 0 as the focus (for async CLI workflows).

---

### Async Mode (Command-line)

If you launch the program with command-line arguments, it will run in non-interactive asynchronous mode:
./Hyperblocks.exe <classIndex>

This mode is used for batch experiments or headless execution on a remote machine or benchmark station.

---
## Project Structure

### `Host.cu`
This is the main entry point of the project. It contains both the `runInteractive` and `runAsync` functions for launching the program. Additionally, it includes utility methods such as `kFold` for various HB modes. While originally intended to only contain entry logic, it currently includes some miscellaneous logic that may later be refactored into dedicated files.

### `CMakeLists.txt`
Defines the build process. CMake simplifies compilation of large projects by handling dependencies and file structures. If you add new source or header files, make sure to update this script to ensure the build includes them.

### `datasets/`
Contains training, testing, and validation datasets. You should place all input `.csv` or preprocessed data files in this directory.

### `classification_testing/`
Contains modules that test and evaluate classifier performance. This includes building confusion matrices, collecting per-point classification metadata, and supporting fallback classifier logic (e.g., precision-weighted voting).

### `data_utilities/`
Houses general-purpose utilities such as normalization, k-fold splitting, and input/output helpers. These functions support the preprocessing and organization of training/testing data.

### `hyperblock/`
Defines the `Hyperblock` data structure and its associated logic. This includes `insideHB` checks, precision calculations, and metadata relevant for classification.

### `hyperblock_generation/`
Contains the CUDA-accelerated logic for generating and simplifying HBs. This includes:

- `mergerHyperBlocks`: CUDA kernel that attempts to merge compatible HBs by expanding their bounds and verifying against opposing class points. Uses float4 vectorization and cooperative evaluation for efficiency.
- `rearrangeSeedQueue`: Reorders the queue of seed blocks after merging, pushing merged blocks to the back to mimic Lincoln Hubers initial algorithm. (May not need to do this, it is an artifact at this point.)
- `assignPointsToBlocks` and `findBetterBlocks`: Functions that assign points to the most appropriate HB, favoring largest blocks for ambiguous cases.
- `removeUselessAttributes`: Prunes dimensions from blocks by testing whether a full-range attribute ([0,1]) would introduce classification errors. Supports disjunctive (multi-interval) representations.


### `interval_hyperblock/`

This includes the first step of the HB generation process. Essentially this file handles the generation of interval HBs. 
These are made by sorting each class by its "best" or "most separating" attribute, then finding PURE intervals in which only that
class exists. Ex. class 5 attribute 1, the interval [0, .5] only contains points from class 5, then this would be made into 
a pure interval HB. For more information: https://arxiv.org/abs/2506.06986.


### `knn/`
Implements several fallback k-Nearest Neighbor algorithms. Includes:
- pure k-NN on raw data
- HB-guided k-NN
- precision-weighted voting
- threshold-based classification  
  These are used when HBs alone cannot classify a point.

### `lda/`
Provides a multiclass Linear Discriminant Analysis (LDA) implementation. This file is currently only used for trying to find
a more optimal removal order for removeUselessAttributes/removeRedundantAttributes.

### `screen_output/`
Provides utilities for displaying results and interacting with the command-line interface. Includes:

- `displayMainMenu`: Prints the interactive menu used in `Host.cu`.
- `clearScreen` and `waitForEnter`: Cross-platform terminal controls.
- `printConfusionMatrix`: Displays the confusion matrix and calculates per-class and overall performance metrics (accuracy, precision, recall, F1) via `computePerformanceMetrics`.
- `printDataset`: Outputs full dataset contents for debugging.

Used for evaluation output, debugging, and user-driven menu interaction.

### `simplifications/`
Implements iterative HB simplification routines. Includes:

- `removeUselessBlocks`: CUDA-accelerated routine that eliminates redundant blocks based on point coverage frequency.
- `removeUselessAttr`: Identifies and removes attributes from blocks if they do not contribute to class separation.
- `runSimplifications`: Combines both techniques and applies them until no further block or attribute pruning is possible.

We detail some simplification algorithms in this paper: https://arxiv.org/abs/2506.06986. 
  
## Contact Information & Credits

This project was developed at the Central Washington University VKD Lab under the mentorship of Dr. Boris Kovalerchuk, and is based on the DV2.0 Hyperblocks model.

### Contributors

---

- **Austin Snyder**  
  - School email: [austin.snyder@cwu.edu](mailto:austin.snyder@cwu.edu)
  - Personal email: [austin.w.snyder@outlook.com](mailto:austin.w.snyder@outlook.com)
  - LinkedIn: [linkedin.com/in/austinsnyder411](https://www.linkedin.com/in/austinsnyder411/)
  - Discord: mxstic.  

- **Ryan Gallagher**  
  - Email: [ryan.gallagher@cwu.edu](mailto:ryan.gallagher@cwu.edu)  
  - LinkedIn: [linkedin.com/in/ryan-gallagher-0b2095285](https://www.linkedin.com/in/ryan-gallagher-0b2095285/)