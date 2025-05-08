# HyperBlocks
Standalone version of CWU-VKD-LAB DV Hyper-Blocks functionality. The aim is to translate host code from Java to C++ for use on CWU CUDA Cluster. 

Hyper-Blocks are a rule based Machine Learning model. They include a minimum and maximum value
for all attributes in the dataset. This allows for a clear graphical representation of 
the classifier through the use of Parallel Coordinates. Easily interpretable graphical representations
allow for Subject-Matter-Experts to analyze the decision-making of models and increase trust
when compared to Black-Box models.


Compile:
nvcc -Xcompiler /openmp -o a ./Host.cu ./hyperblock/HyperBlock.cpp ./hyperblock_generation/MergerHyperBlock.cu ./cuda_util/CudaUtil.cpp ./data_utilities/DataUtil.cpp ./interval_hyperblock/IntervalHyperBlock.cu ./knn/Knn.cpp ./screen_output/PrintingUtil.cpp ./simplifications/Simplifications.cu ./ClassificationTesting/ClassificationTests.cpp -g -G -O3 

Run:
    ./a
    or
    ./a.exe