/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Venny
 */
public class MyC45 {
    
    
    // Computes the entropy of a dataset.
    private double computeEntropy(Instances data) throws Exception {
        double [] labelCounter = new double[data.numClasses()];
        for(int i=0; i<data.numInstances(); ++i){
            labelCounter[(int) data.instance(i).classValue()]++;
        }
        
        double entropy = 0;
        for (int i=0; i<labelCounter.length; ++i) {
            if (labelCounter[i] > 0) {
                double proportion = labelCounter[i]/data.numInstances();
                entropy -= (proportion)*Utils.log2(proportion);
            }
        }
        return entropy;
    }
    
    // Splits a dataset according to the values of a nominal attribute.
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        for (int i=0; i<data.numInstances(); ++i) {
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        }

        for (int i=0; i<splitData.length; ++i) {
            splitData[i].compactify();
        }
        return splitData;
    }
    
    // Computes information gain for an attribute.
    private double computeIG(Instances data, Attribute att) throws Exception {
        double IG = computeEntropy(data);
        if (att.isNumeric()){
            
        } else { // nominal
            Instances[] splitData = splitData(data, att);
            for (Instances splitdata : splitData) {
                if (splitdata.numInstances() > 0) {
                    double splitNumInstances = splitdata.numInstances();
                    double dataNumInstances = data.numInstances();
                    double proportion = splitNumInstances / dataNumInstances;
                    IG -= proportion * computeEntropy(splitdata);
                }
            }
        }
        return IG;
    }
    
    // Calculate threshold for numeric attributes
    private double caculateThreshold(Instances data, Attribute att) {
        double threshold = 0;
        // sort data berdasarkan attr
        // foreach instance
            // if class current instance beda sama class next instance
                // hitung information gain
                // if IG > threshold
                    // threshold = current attr value
        return threshold;
    }
}
