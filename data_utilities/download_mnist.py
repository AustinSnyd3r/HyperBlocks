from sklearn.datasets import fetch_openml
import numpy as np
import csv

def download_mnist():
    # Download MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize to [0,1] range
    X = X / 255.0
    
    # Split into train (60000) and test (10000)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Save training data
    with open('../datasets/mnist_train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(X_train)):
            row = list(X_train[i]) + [y_train[i]]
            writer.writerow(row)
    
    # Save test data
    with open('../datasets/mnist_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(X_test)):
            row = list(X_test[i]) + [y_test[i]]
            writer.writerow(row)
    
    print(f"Saved {len(X_train)} training samples to mnist_train.csv")
    print(f"Saved {len(X_test)} test samples to mnist_test.csv")

if __name__ == '__main__':
    download_mnist()