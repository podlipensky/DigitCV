//
//  main.cpp
//  DigitCV
//
//  Created by Pavlo Pidlypenskyi on 1/3/16.
//  Copyright © 2016 Pavlo Pidlypenskyi. All rights reserved.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "NetworkSGD.cpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    // insert code here...
    unsigned top[] = {784, 100, 10};
    vector<unsigned> topology(top, top+sizeof(top)/sizeof(top[0]));
    NetworkSGD net = NetworkSGD(topology);
    
    // read train data
    string fileName = "/Users/podlipensky/sites/digits-data/train.csv";
    Ptr<ml::TrainData> raw_data = ml::TrainData::loadFromCSV(fileName, 785, 0, 1);
    if (raw_data.empty()) {
        printf("ERROR: File %s can not be read\n", fileName.c_str());
        return 0;
    }
    // train network
    net.Train(raw_data, 20, 10, 3.0, 0.1);
    return 0;
}
