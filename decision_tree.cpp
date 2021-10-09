#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

template<typename T>
vector<T> createThresholds(T start, T stop, int n) {
    float epsilon = 0.0005f;
    T step = (stop - start) / n;
    vector<T> values;

    // Skip first and last thresholds
    T value = start + step;
    while (value < stop - step) {
        values.push_back(value);
        value += step;
    }
    return values;
}

class Node {
private:
    double gini;
    int numSamples;
    vector<int> numSamplesPerClass;
    int predictedClass;
    Node *left;
    Node *right;
    int featureIndex;
    double featureThreshold;
public:
    Node(double gini, int numSamples, vector<int> numSamplesPerClass, int predictedClass) {
        this->gini = gini;
        this->numSamples = numSamples;
        this->numSamplesPerClass = numSamplesPerClass;
        this->predictedClass = predictedClass;
        this->left = nullptr;
        this->right = nullptr;
        this->featureIndex = -1;
        this->featureThreshold = -1;
    }

    void setFeatureIndex(int featureIndex_) {
        this->featureIndex = featureIndex_;
    }

    void setFeatureThreshold(double featureThreshold_) {
        this->featureThreshold = featureThreshold_;
    }

    void setLeftNode(Node *node) {
        this->left = node;
    }

    void setRightNode(Node *node) {
        this->right = node;
    }

    Node *getLeftNode() {
        return this->left;
    }

    Node *getRightNode() {
        return this->right;
    }

    int getPredictedClass() const {
        return this->predictedClass;
    }

    int getFeatureIndex() const {
        return this->featureIndex;
    }

    double getFeatureThreshold() const {
        return this->featureThreshold;
    }
};

class DecisionTree {
private:
    int numSamples;
    int numFeatures;
    int numClasses;
    Node *root;
    string criterion;
    int maxDepth;

    /**
     * Given a vector of classes, we compute the number of unique classes.
     */
    int calculateNumClasses(const vector<int> &y) {
        unordered_set<int> classes;
        for (auto &c : y) {
            if (classes.find(c) == classes.end()) {
                classes.insert(c);
            }
        }
        return (int) classes.size();
    }

    /**
     * Given a vector classes, we build a vector that contain the number of samples per class.
     */
    vector<int> buildNumSamplesPerClass(const vector<int> &y) {
        vector<int> numSamplesPerClass;
        for (int i = 0; i < numClasses; i++) {
            int counter = 0;
            for (auto &c: y) {
                if (c == i) {
                    counter++;
                }
            }
            numSamplesPerClass.push_back(counter);
        }
        return numSamplesPerClass;
    }

    /**
     * Compute the entropy score for a given vector.
     */
    double computeEntropy(const vector<int> &y) {
        unordered_map<int, int> classFrequencies;

        for (auto &c : y) {
            if (classFrequencies.find(c) == classFrequencies.end()) {
                classFrequencies.insert(make_pair(c, 1));
            } else {
                classFrequencies[c]++;
            }
        }

        double entropy = 0;
        for (auto &it: classFrequencies) {
            double pi = it.second / (double) y.size();
            entropy += pi * log2(pi);
        }

        return -entropy;
    }

    /**
     * Compute the gini score for a given vector.
     */
    double computeGini(const vector<int> &y) {
        unordered_map<int, int> classFrequencies;

        for (auto &c : y) {
            if (classFrequencies.find(c) == classFrequencies.end()) {
                classFrequencies.insert(make_pair(c, 1));
            } else {
                classFrequencies[c]++;
            }
        }

        double giniImpurity = 0;
        for (auto &it: classFrequencies) {
            double pi = it.second / (double) y.size();
            giniImpurity += pi * (1 - pi);
        }

        return giniImpurity;
    }

    double evaluateSplit(const vector<int> &y) {
        if (criterion == "gini") {
            return computeGini(y);
        }
        return computeEntropy(y);
    }

    /**
     * Compute the best split. The first element is the feature index and the second element is the feature threshold.
     */
    pair<int, double> computeBestSplit(const vector<vector<double>> &X, const vector<int> &y) {
        if (y.empty()) {
            return make_pair(-1, -1);
        }

        vector<int> numSamplesPerClass = buildNumSamplesPerClass(y);
        double initialSplitScore = evaluateSplit(y);
        double bestInformationGain = 0;
        int bestFeatureIndex = 0;
        double bestFeatureThreshold = 0;
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            vector<double> feature;
            feature.reserve(X.size());
            for (const auto &x : X) {
                feature.push_back(x[featureIndex]);
            }

            double featureMin = *min_element(feature.begin(), feature.end());
            double featureMax = *max_element(feature.begin(), feature.end());
            vector<double> thresholds = createThresholds(featureMin, featureMax, 5);
            for (auto &threshold : thresholds) {
                vector<int> yLeft;
                vector<int> yRight;

                for (int i = 0; i < feature.size(); i++) {
                    if (feature[i] < threshold) {
                        yLeft.push_back(y[i]);
                    } else {
                        yRight.push_back(y[i]);
                    }
                }

                int yLeftSize = (int) yLeft.size();
                int yRightSize = (int) yRight.size();
                double leftScore = evaluateSplit(yLeft);
                double rightScore = evaluateSplit(yRight);
                double splitScore = ((yLeftSize) * leftScore + (yRightSize) * rightScore) / (yLeftSize + yRightSize);
                double splitInformationGain = initialSplitScore - splitScore;
                if (splitInformationGain > bestInformationGain) {
                    bestInformationGain = splitInformationGain;
                    bestFeatureIndex = featureIndex;
                    bestFeatureThreshold = threshold;
                }
            }
        }

        return make_pair(bestFeatureIndex, bestFeatureThreshold);
    }

    Node *buildTree(const vector<vector<double>> &X, const vector<int> &y, int depth = 0) {
        vector<int> numSamplesPerClass = buildNumSamplesPerClass(y);
        int predictedClass =
                max_element(numSamplesPerClass.begin(), numSamplesPerClass.end()) - numSamplesPerClass.begin();
        double gini = computeGini(y);
        Node *node = new Node(gini, (int) y.size(), numSamplesPerClass, predictedClass);

        if (depth < this->maxDepth) {
            pair<int, double> bestSplit = computeBestSplit(X, y);
            int featureIndex = bestSplit.first;
            double featureThreshold = bestSplit.second;
            if (featureIndex != -1) {
                vector<bool> leftIndexes;
                for (const auto &row : X) {
                    if (row[featureIndex] < featureThreshold) {
                        leftIndexes.push_back(true);
                    } else {
                        leftIndexes.push_back(false);
                    }
                }

                vector<vector<double> > xLeft, xRight;
                vector<int> yLeft, yRight;
                for (int i = 0; i < leftIndexes.size(); i++) {
                    vector<double> v;
                    copy(X[i].begin(), X[i].end(), back_inserter(v));
                    if (leftIndexes[i]) {
                        xLeft.push_back(v);
                        yLeft.push_back(y[i]);
                    } else {
                        xRight.push_back(v);
                        yRight.push_back(y[i]);
                    }
                }

                node->setFeatureIndex(featureIndex);
                node->setFeatureThreshold(featureThreshold);
                node->setLeftNode(buildTree(xLeft, yLeft, depth + 1));
                node->setRightNode(buildTree(xRight, yRight, depth + 1));
            }
        }

        return node;
    }

    int predictRow(vector<double> x) {
        Node *node = this->root;
        while (node->getLeftNode()) {
            if (x[node->getFeatureIndex()] < node->getFeatureThreshold()) {
                node = node->getLeftNode();
            } else {
                node = node->getRightNode();
            }
        }
        return node->getPredictedClass();
    }

public:
    DecisionTree(const string &criterion, int maxDepth = -1) {
        this->criterion = criterion;
        this->maxDepth = maxDepth;
    }

    void fit(const vector<vector<double>> &X, const vector<int> &y) {
        this->numSamples = (int) X.size();
        this->numFeatures = (int) X[0].size();
        this->numClasses = calculateNumClasses(y);
        this->root = buildTree(X, y);
    }

    vector<int> predict(const vector<vector<double >> &X) {
        vector<int> predictions;
        predictions.reserve(X.size());
        for (auto &x: X) {
            predictions.push_back(this->predictRow(x));
        }
        return predictions;
    }
};

void trainTestSplit(vector<vector<double>> X, vector<int> Y,
                    vector<vector<double>> &X_train, vector<int> &y_train,
                    vector<vector<double>> &X_test, vector<int> &y_test,
                    double trainPerc, double testPerc) {
    int trainN = trainPerc * X.size();
    int testN = testPerc * X.size();

    cout << "Empezando Train y Test split con " << trainN << " y " << testN << " valores respectivamente." << endl;
    vector<int> index;
    index.reserve(X.size());
    for (int i = 0; i < X.size(); ++i) index.push_back(i);
    random_shuffle(index.begin(), index.end());
    for (int i = 0; i < trainN; i++) {
        X_train.push_back(X.at(index.at(i)));
        y_train.push_back(Y.at(index.at(i)));
    }
    for (int i = trainN; i < trainN + testN; i++) {
        X_test.push_back(X.at(index.at(i)));
        y_test.push_back(Y.at(index.at(i)));
    }
    cout << "Se obtuvieron Train (X,Y): " << X_train.size() << " y " << y_train.size() << " registros ." << endl;
    cout << "Se obtuvieron Test (X,Y): " << X_test.size() << " y " << y_test.size() << " registros ." << endl;
    cout << endl;
}

pair<vector<vector<double> >, vector<int> > readIrisCSV() {
    string path = "iris.csv";
    char separator = ',';
    ifstream file(path);
    vector<vector<double> > X;
    vector<int> y;

    string line;
    getline(file, line); // skip first line
    while (file) {
        if (!getline(file, line)) {
            break;
        }

        stringstream ss(line);
        vector<double> row;
        while (ss) {
            string s;
            if (!getline(ss, s, separator)) {
                break;
            }
            if (ss.rdbuf()->in_avail() > 0) {
                row.push_back(stod(s));
            } else {
                if (s == "\"Setosa\"") {
                    y.push_back(0);
                } else if (s == "\"Versicolor\"") {
                    y.push_back(1);
                } else if (s == "\"Virginica\"") {
                    y.push_back(2);
                }
            }
        }
        X.push_back(row);
    }
    return make_pair(X, y);
}

void confusionMatrix(vector<int> Y_test, vector<int> Y_pred, vector<int> &classes) {
    cout << "Empezando cálculo de Matriz de confusión para " << classes.size() << " clases." << endl;
    vector<vector<int>> confusionM;
    for (int i = 0; i < classes.size(); ++i) {
        vector<int> linea;
        linea.reserve(classes.size());
        for (int j = 0; j < classes.size(); ++j) linea.push_back(0);
        confusionM.push_back(linea);
    }
    int cFPFN = 0;
    int cTPTN = 0;
    for (int i = 0; i < Y_test.size(); ++i) {
        if (Y_test.at(i) != Y_pred.at(i)) cFPFN++;
        else if (Y_test.at(i) == Y_pred.at(i))cTPTN++;
        confusionM.at(Y_test.at(i)).at(Y_pred.at(i)) += 1;

    }
    cout << "Finalizo la creación de matriz de confusión con los siguientes resultados:" << endl;
    for (int i = 0; i < confusionM.size(); i++) {
        cout << "  " << classes.at(i) << "";
    }
    cout << endl;
    for (int i = 0; i < confusionM.size(); i++) {
        cout << classes.at(i) << "[";
        for (int j = 0; j < confusionM.at(i).size(); j++) {
            cout << confusionM.at(i).at(j);
            if (j < confusionM.at(i).size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
    cout << endl;
    cout << "TP or TN:" << cTPTN << endl;
    cout << "FP or FN:" << cFPFN << endl;
    cout << endl;
}

void evaluateDecisionTree(const vector<vector<double>> &X_train, const vector<int> &y_train,
                          const vector<vector<double>> &X_test, const vector<int> &y_test, const string &criterion) {
    cout << "Evaluate DecisionTree using " << criterion << endl;
    DecisionTree *dt = new DecisionTree(criterion, 3);
    dt->fit(X_train, y_train);
    vector<int> y_pred = dt->predict(X_test);

    vector<int> classes = {0, 1, 2};
    confusionMatrix(y_test, y_pred, classes);

}

int main() {
    auto dataset = readIrisCSV();
    vector<vector<double> > X = dataset.first;
    vector<int> y = dataset.second;
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;
    trainTestSplit(X, y, X_train, y_train, X_test, y_test, 0.7, 0.3);

    evaluateDecisionTree(X_train, y_train, X_test, y_test, "gini");
    evaluateDecisionTree(X_train, y_train, X_test, y_test, "entropy");

    return 0;
}
