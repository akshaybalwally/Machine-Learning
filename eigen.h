
#ifndef EIGEN_H
#define EIGEN_H
#define E 2.71828
#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <time.h>

// next line needed because NULL is part of std namespace
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class eigen {
public:
    eigen(int numF, int numH, int numP);                 // Constructor
    void addProperty(VectorXd property);
    void initializeThetas();
    VectorXd genHouse();
    VectorXd genBoat();
    void populate();
    void normalizeProperty(VectorXd * property);
    double sigmoid(double value);
    void sigmoid(VectorXd * values);
    double calculate(VectorXd property);
    void calculateAll();
    double run();
    void backPropogate(int repetitions);
    VectorXd elementWiseProduct(VectorXd first, VectorXd second);
    void setFinalTheta(VectorXd vec);
    void updateThetas(vector<MatrixXd> * dThetas, VectorXd dFinal);
    void test(int testSize);

private:
    VectorXd finalTheta;
    int population;
    vector<VectorXd> * properties;
    vector<int> * propertyTypes;
    int numFeatures;
    int numHiddenLayers;
    vector<MatrixXd> * thetas;
    VectorXd uno;
    VectorXd zero;
    double alpha;
    double lambda;
};

#endif
