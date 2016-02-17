//
//  NetworkSGD.cpp
//  DigitCV
//
//  Created by Pavlo Pidlypenskyi on 1/8/16.
//  Copyright © 2016 Pavlo Pidlypenskyi. All rights reserved.
//

#include "NetworkSGD.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

class NetworkSGD {
    
private:
    unsigned m_layersCount;
    vector<unsigned> m_sizes;
    vector<Mat> m_biases;
    vector<Mat> m_weights;
    
    Mat GetRandomMatOfSize(int n, int m) {
        int sz[2] = { n, m };
        Mat b(2, sz, CV_32F);
        randu(b, -3.0f, 3.0f);
        return b;
    }
    
public:
    NetworkSGD(const vector<unsigned> &sizes) {
        m_layersCount = sizes.size();
        m_sizes = sizes;
        for (int i = 1; i < m_layersCount; i++) {
            m_biases.push_back(GetRandomMatOfSize((int)m_sizes[i], 1));
            m_weights.push_back(GetRandomMatOfSize((int)m_sizes[i], (int)m_sizes[i-1]));
        }
    }
    
    void Train(const Ptr<ml::TrainData> &data, unsigned epochCount, unsigned batchSize) {
        // Looks like there is a bug in OpenCV implementation of shuffleTrainTest since it
        // mixes responses in incorrect way. Therefore the algorithm is less efficient because
        // we train it based on the same dataset over and over again.
        // data->shuffleTrainTest();
        Mat samples = data->getTrainSamples();
        samples = samples/255.0;
        Mat labels = data->getTrainResponses();
        double testTrainRatio = 0.1;
        int trainRowCount = (int)samples.rows * (1-testTrainRatio);
        int testRowCount = (int)samples.rows * testTrainRatio;
        
        Mat trainSamples = samples.rowRange(cv::Range(0, trainRowCount));
        Mat trainLabels = labels.rowRange(cv::Range(0, trainRowCount));
        Mat testSamples = samples.rowRange(cv::Range(trainRowCount, trainRowCount+testRowCount));
        Mat testLabels = labels.rowRange(cv::Range(trainRowCount, trainRowCount+testRowCount));
        
        for (unsigned e = 0; e < epochCount; e++) {
            cout << "Start epoch " << e << endl;
            for(unsigned i = 0; i < trainSamples.rows-batchSize; i+=batchSize) {
                UpdateMiniBatch(trainSamples.rowRange(cv::Range(i, i+batchSize)),
                                trainLabels.rowRange(cv::Range(i, i+batchSize)),
                                3.0);
            }
            Evaluate(e, testSamples, testLabels);
        }
    }

    Mat FeedForward(const Mat &input) {
        Mat activation = input.t();
        for(int i = 0; i < m_weights.size(); i++) {
            activation = Sigmoid(m_weights[i]*activation + m_biases[i]);
        }
        return activation;
    }
    
    void Evaluate(unsigned epoch, const Mat &samples, const Mat &labels) {
        unsigned correctCount = 0;
        for(int i = 0; i < samples.rows; i++) {
            Mat output = FeedForward(samples.row(i));
            // find index of the max element
            int maxIdx = -1;
            double max = INT_MIN;
            for(int i = 0; i < output.rows; i++) {
                if (output.at<float>(i, 0) > max) {
                    max = output.at<float>(i, 0);
                    maxIdx = i;
                }
            }
            if (maxIdx == (int)labels.at<float>(i, 0)) {
                correctCount++;
            }
        }
        cout << "Epoch " << epoch << ": got " << correctCount << " out of " << samples.rows << endl;
    }
    
    void UpdateMiniBatch(Mat batch, Mat batch_labels, double eta) {
        vector<Mat> nabla_b = GetBiasesMatZeros();
        vector<Mat> nabla_w = GetWeightsMatZeros();
        
        for (int i = 0; i < batch.rows; i++) {
            Mat X = batch.row(i);
            Mat y = LabelToMat(batch_labels.at<float>(i, 0));
            vector<Mat> delta_nabla_b = GetBiasesMatZeros();
            vector<Mat> delta_nabla_w = GetWeightsMatZeros();
            
            // Calculate derivatives dC/db and dC/dw
            Backprop(X, y, delta_nabla_b, delta_nabla_w);
            
            vector<Mat>::iterator nw_it = nabla_w.begin();
            vector<Mat>::iterator dnw_it = delta_nabla_w.begin();
            vector<Mat>::iterator nb_it = nabla_b.begin();
            vector<Mat>::iterator dnb_it = delta_nabla_b.begin();
            for(; nw_it != nabla_w.end(); ++nw_it, ++dnw_it, ++nb_it, ++dnb_it) {
                *nw_it += *dnw_it;
                *nb_it += *dnb_it;
            }
        }
        for (int i = 0; i < m_layersCount-1; i++) {
            m_weights[i] = m_weights[i] - (eta/batch.rows) * nabla_w[i];
            m_biases[i] = m_biases[i] - (eta/batch.rows) * nabla_b[i];
        }
    }
    
    void Backprop(const Mat &X, const Mat &y, vector<Mat> &delta_b, vector<Mat> &delta_w) {
        // Calculate delta_b and delta_w representing the graident for the
        // cost function C_x
        Mat activation = X.t();
        // vector to store all the activations, layer by layer
        vector<Mat> activations;
        activations.push_back(activation);
        // vector to store all the z vectors, layer by layer
        vector<Mat> zs;
        // forward pass
        for (int i = 0; i < m_weights.size(); i++) {
            Mat w = m_weights[i];
            Mat b = m_biases[i];
            Mat z = w*activation+b;
            zs.push_back(z);
            activation = Sigmoid(z);
            activations.push_back(activation);
        }
        // backward pass
        Mat delta = CostDerivative(activations.back(), y).mul(SigmoidPrime(zs.back()));
        int lastIdx = delta_b.size()-1;
        delta_b[lastIdx] = delta;
        delta_w[lastIdx] = delta*activations[m_layersCount-2].t();
        
        for (int l = 2; l < m_layersCount; l++) {
            int i = m_layersCount - l - 1;  // substract -1 because we one layer short in weights
            Mat z = zs[i];
            Mat sp = SigmoidPrime(z);
            delta = (m_weights[i+1].t() * delta).mul(sp);
            delta_b[i] = delta;
            delta_w[i] = delta * activations[i].t();
        }
    }
    
    vector<Mat> GetBiasesMatZeros() {
        vector<Mat> nabla_b(m_biases.size());
        for (int i = 0; i < m_biases.size(); i++) {
            nabla_b[i] = Mat::zeros(m_biases[i].rows, m_biases[i].cols, CV_32F);
        }
        return nabla_b;
    }
    
    vector<Mat> GetWeightsMatZeros() {
        vector<Mat> nabla_w(m_weights.size());
        for (int i = 0; i < m_weights.size(); i++) {
            nabla_w[i] = Mat::zeros(m_weights[i].rows, m_weights[i].cols, CV_32F);
        }
        return nabla_w;
    }
    
    Mat LabelToMat(float label) {
        // convert float label into Mat
        Mat labelMat = Mat::zeros(10, 1, CV_32F);
        labelMat.at<float>((int)label, 0) = 1.0;
        return labelMat;
    }
    
    Mat CostDerivative(const Mat outputActivations, const Mat target) {
        return outputActivations - target;
    }
    
    Mat Sigmoid(const Mat &z) {
        Mat sz(z);
        cv::exp(-z, sz);
        return 1.0 / (1.0 + sz);
    }
    
    Mat SigmoidPrime(const Mat &z) {
        return Sigmoid(z).mul(1-Sigmoid(z));
    }
};

















