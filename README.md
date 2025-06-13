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

Hyperblocks are an interpretable, rule-based machine learning model. Each hyperblock defines axis-aligned bounds (min/max) for each attribute in the dataset, forming a hyper-rectangle in feature space.

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
nvcc -Xcompiler /openmp -o a.exe ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./ClassificationTesting/ClassificationTests.cpp -g -G -O3
```

- **Run**:
```bash
a.exe
```

### Linux

- **Compile**:
```bash
nvcc -Xcompiler -fopenmp -o a ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./ClassificationTesting/ClassificationTests.cpp -g -G -O3
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
- The **last column** must be the **class label** as an integer (e.g., 0, 1, 2...).

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
- If the Hyperblock save file and dataset do not align in number of attributes, the program **may silently fail or misclassify**.
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
- Generate new HyperBlocks
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
  - Case 6 generates Interval HBs
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