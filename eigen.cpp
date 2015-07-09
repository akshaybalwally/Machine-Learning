#include "eigen.h"

eigen::eigen(int numF, int numH, int numP) {
	srand(time(NULL));
	numFeatures = numF;
	numHiddenLayers = numH;
	population = numP;
	properties = new vector<VectorXd>();
	propertyTypes = new vector<int>();
	thetas = new vector<MatrixXd>();
	alpha = 4;
	lambda = 0.01;


	for (int i = 0; i < numHiddenLayers; i++) {
		MatrixXd temptheta(numFeatures, numFeatures + 1);
		thetas->push_back(temptheta);
	}
	VectorXd finalT(numFeatures + 1);
	for (int i = 0; i < numFeatures + 1; i++) {
		finalT(i) = (double)rand() / (RAND_MAX / 2) - 1;
	}
	finalTheta = finalT;


	VectorXd tuno(numFeatures + 1, 1);
	VectorXd tzero(numFeatures + 1, 1);

	for (int loc = 0; loc < numFeatures + 1; loc++) {
		tuno(loc) = 1;
		tzero(loc) = 0;
	}
	uno = tuno;
	zero = tzero;

}

VectorXd eigen::elementWiseProduct(VectorXd first, VectorXd second) {
	if (!(first.cols() == second.cols() && first.rows() == second.rows())) {
		cout << "Invalid operation: different sizes" << endl;
		VectorXd vec(1);
		return vec;
	}
	VectorXd retVal(first.size());
	for (int i = 0; i < retVal.size(); i++) {
		retVal(i) = first(i) * second(i);
	}
	return retVal;
}

double eigen::run() {
	double unalteredCost = 0;
	for (int i = 0; i < properties->size(); i ++) {
		double h = calculate(properties->at(i));
		unalteredCost = unalteredCost + propertyTypes->at(i) * log(h) + (1 - propertyTypes->at(i)) * log(1 - h);
	}
	unalteredCost = unalteredCost / properties->size();
	unalteredCost = unalteredCost * -1;
	cout << "Unaltered Cost is: " << unalteredCost << endl;
	double regularization = 0;
	for (int i = 0 ; i < thetas->size(); i++) {
		for (int j = 1; j < thetas->at(i).rows(); j++) {
			for (int k = 0; k < thetas->at(i).cols(); k++) {
				regularization = regularization + pow((thetas->at(i))(j, k), 2);
			}
		}
	}
	for (int i = 1; i < finalTheta.size(); i++) {
		regularization = regularization + pow(finalTheta(i), 2);
	}
	double lambda = 1;
	regularization = regularization * lambda / (2 * properties->size());
	cout << thetas->at(0) << endl;
	cout << "regularization Cost is: " << regularization << endl;
	return regularization + unalteredCost;
}

void eigen::setFinalTheta(VectorXd vec) {
	finalTheta = vec;
}

void eigen::updateThetas(vector<MatrixXd> * dThetas, VectorXd dFinal) {
	for (int i = 0; i < numHiddenLayers; i++) {
		thetas->at(i) = thetas->at(i) - dThetas->at(i) * alpha;
	}
	setFinalTheta(finalTheta - alpha * dFinal);

}

void eigen::backPropogate(int repetitions) {

	for (int reps = 0; reps < repetitions; reps++) {
		double totalHmiss = 0;
		vector<MatrixXd> * dThetaAccumulators = new vector<MatrixXd>();
		for (int i = 0; i < numHiddenLayers; i++) {
			MatrixXd dtemptheta(numFeatures, numFeatures + 1);
			dThetaAccumulators->push_back(dtemptheta);
		}
		VectorXd dFinalThetaAccumulator(numFeatures + 1);

		vector<MatrixXd> * dThetas = new vector<MatrixXd>();
		for (int i = 0; i < numHiddenLayers; i++) {
			MatrixXd dtemptheta(numFeatures, numFeatures + 1);
			dThetas->push_back(dtemptheta);
		}
		VectorXd dFinalTheta(numFeatures + 1);

		double unalteredCost = 0;
		for (int prop = 0; prop < properties->size(); prop++) {
			VectorXd vec(numFeatures);
			vec = properties->at(prop);
			MatrixXd mat(numFeatures + 1, numHiddenLayers + 1);

			for (int i = 0; i < numHiddenLayers + 1; i++) {

				mat(0, i) = 1;

				mat.block(1, i, numFeatures, 1) = vec;

				if (i < numHiddenLayers) {

					vec = thetas->at(i) * mat.col(i);

					sigmoid(&vec);

				}
			}

			//cout << mat << endl;

			double h = finalTheta.dot(mat.col(numHiddenLayers));

			h = sigmoid(h);

			double tempCost = propertyTypes->at(prop) * log(h) + (1 - propertyTypes->at(prop)) * log(1 - h);

			unalteredCost = unalteredCost + tempCost;

			double hMiss = h - propertyTypes->at(prop);

			totalHmiss = totalHmiss + abs(hMiss);

			//BackPropogate HERE

			//Note that errors' first column will be 0, representing 0 error on the inputs

			MatrixXd errors(numFeatures + 1, numHiddenLayers + 1);
			for (int layer = numHiddenLayers; layer > 0; layer--) {
				VectorXd S(numFeatures + 1, 1);
				VectorXd aVals(numFeatures + 1);
				aVals = mat.col(layer);
				S = elementWiseProduct(finalTheta.transpose() * hMiss, elementWiseProduct(aVals, (uno - aVals)));
				
				errors.col(layer) = S;
			}
			errors.col(0) = zero;
			for (int thetaLoc = 0; thetaLoc < numHiddenLayers; thetaLoc++) {
				MatrixXd tempMat(numFeatures + 1, numFeatures + 1);
				tempMat = errors.col(thetaLoc + 1) * mat.col(thetaLoc).transpose();

				MatrixXd thetaChange(numFeatures, numFeatures + 1);
				thetaChange = tempMat.block(1, 0, numFeatures, numFeatures + 1);

				dThetaAccumulators->at(thetaLoc) = dThetaAccumulators->at(thetaLoc) + thetaChange;

			}

			//cout << "hMiss is: " << hMiss << endl;

			dFinalThetaAccumulator = dFinalThetaAccumulator + hMiss * mat.col(2);


		}

		for (int thet = 0; thet < numHiddenLayers; thet++) {
			for (int rows = 0; rows < numFeatures; rows++) {
				for (int cols = 0; cols < numFeatures + 1; cols++) {
					if (cols != 0) {
						(dThetas->at(thet))(rows, cols) = (dThetaAccumulators->at(thet))(rows, cols) / properties->size() + lambda * (thetas->at(thet))(rows, cols);

					}
					else {
						(dThetas->at(thet))(rows, cols) = (dThetaAccumulators->at(thet))(rows, cols) / properties->size();

					}
				}
			}

		}
		for (int element = 0; element < dFinalTheta.size(); element++) {
			if (element == 0) {
				dFinalTheta(element) = dFinalThetaAccumulator(element) / properties->size();
			}
			else {
				dFinalTheta(element) = dFinalThetaAccumulator(element) / properties->size() + lambda * finalTheta(element);
			}

		}


		updateThetas(dThetas, dFinalTheta);

		//Prints to show whats been done

		// for (int prints = 0; prints < numHiddenLayers; prints++) {
		// 	cout << "Layer: " << prints << " is: " << endl << dThetaAccumulators->at(prints) << endl;
		// }
		// cout << "dFinalThetaAccumulator is " << dFinalThetaAccumulator << endl;

		cout << "TotalHMiss is " << totalHmiss << endl;
		double avgHmiss = totalHmiss / properties->size();

		cout << "Avg H Miss: " << avgHmiss << endl;
		//Random cost calculations, have little to do with the actual back-propogation
		unalteredCost = unalteredCost / properties->size();
		unalteredCost = unalteredCost * -1;
		cout << "Unaltered Cost is: " << unalteredCost << endl;
		double regularization = 0;

		for (int i = 0 ; i < thetas->size(); i++) {
			for (int j = 0; j < thetas->at(i).rows(); j++) {
				for (int k = 0; k < thetas->at(i).cols(); k++) {
					if (k != 0) {
						regularization = regularization + pow((thetas->at(i))(j, k), 2);
					}

				}
			}
		}
		for (int i = 1; i < finalTheta.size(); i++) {
			regularization = regularization + pow(finalTheta(i), 2);
		}

		regularization = regularization * lambda / (2 * properties->size());

		cout << "regularization Cost is: " << regularization << endl;
		//cout << "Total Cost: " << regularization + unalteredCost << endl;
		cout << endl << endl << endl;
	}


}



void eigen::addProperty(VectorXd property) {
	properties->push_back(property);
}

void eigen::initializeThetas() {
	for (int i = 0; i < numFeatures; i++) {

		for (int j = 0; j < numFeatures + 1; j++) {
			for (int t = 0; t < thetas->size(); t++) {
				thetas->at(t)(i, j) = (double)rand() / (RAND_MAX / 2) - 1;
			}
		}
	}
}

VectorXd eigen::genHouse() {
	VectorXd house(5);
	house(0) = 150000 + rand() % 200000;
	house(1) = 160000 + rand() % 150000;
	house(2) = 120000 + rand() % 350000;
	house(3) = rand() % 14 + 2;
	house(4) = rand() % 15000 + 8000;
	return house;
}

VectorXd eigen::genBoat() {
	VectorXd boat(5);
	boat(0) = 15000 + rand() % 20000;
	boat(1) = 16000 + rand() % 15000;
	boat(2) = 12000 + rand() % 35000;
	boat(3) = rand() % 4 + 2;
	boat(4) = rand() % 1500 + 800;
	return boat;
}

void eigen::calculateAll() {
	for (int i = 0; i < properties->size(); i ++) {
		calculate(properties->at(i));
	}
}

double eigen::calculate(VectorXd vec) {

	cout << endl << endl << endl;

	MatrixXd mat(numFeatures + 1, numHiddenLayers + 1);

	for (int i = 0; i < numHiddenLayers + 1; i++) {

		mat(0, i) = 1;

		mat.block(1, i, numFeatures, 1) = vec;

		if (i < numHiddenLayers) {

			vec = thetas->at(i) * mat.col(i);

			sigmoid(&vec);

		}
	}


	double result = finalTheta.dot(mat.col(numHiddenLayers));

	result = sigmoid(result);

	return result;

}

void eigen::test(int testSize) {
	vector<VectorXd> * testProperties = new vector<VectorXd>();
	vector<int> * testPropertyTypes = new vector<int>();
	for (int i = 0; i < testSize; i++) {
		int type = rand() % 2;
		VectorXd prop(numFeatures);
		if (type == 1) {
			prop = genBoat();
			normalizeProperty(&prop);
			testProperties->push_back(prop);
			testPropertyTypes->push_back(1);
		}
		else {
			prop = genHouse();
			normalizeProperty(&prop);
			addProperty(prop);
			testProperties->push_back(prop);
			testPropertyTypes->push_back(0);
		}
	}
	for (int i = 0; i < testProperties->size(); i++) {
		double h = calculate(testProperties->at(i));
		cout << "Guessed, Actual: " << h << "   " << testPropertyTypes->at(i) << endl;
	}
}

void eigen::populate() {
	for (int i = 0; i < population; i++) {
		int type = rand() % 2;
		VectorXd prop(numFeatures);
		if (type == 1) {
			prop = genBoat();
			normalizeProperty(&prop);
			addProperty(prop);
			propertyTypes->push_back(1);
		}
		else {
			prop = genHouse();
			normalizeProperty(&prop);
			addProperty(prop);
			propertyTypes->push_back(0);
		}
	}
}
void eigen::normalizeProperty(VectorXd * property) {
	(*property)(0) = (*property)(0) / 350000;
	(*property)(1) = (*property)(1) / 310000;
	(*property)(2) = (*property)(2) / 470000;
	(*property)(3) = (*property)(3) / 16;
	(*property)(4) = (*property)(4) / 23000;
}

double eigen::sigmoid(double value) {
	return 1 / (1 + pow(E, -value));
}

void eigen::sigmoid(VectorXd * values) {
	for (int i = 0; i < (*values).size(); i++) {
		(*values)(i) = sigmoid((*values)(i));
	}
}
