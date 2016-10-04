/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.core.Attribute;
import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Pipin
 */
public class MyID3 extends Classifier {
    private Id3[] successors; //node's successor
    private double label; //class value if node is leaf
    private Attribute attribute; //used for splitting
    private Attribute classAttr; //class attribute of dataset
    private double[] distribution; //class distribution for each label
    
     //Computes the entropy of a dataset.
    private double computeEntropy(Instances data) throws Exception {
        double [] classCounter = new double[data.numClasses()];
        for(int i=0; i<data.numInstances(); ++i){
            Instance inst = (Instance) data.instance(i);
            classCounter[(int) inst.classValue()]++;
        }
        
        double entropy = 0;
        for (int i=0; i<data.numClasses(); ++i) {
            if (classCounter[i] > 0) {
                entropy -= classCounter[i]*Utils.log2(classCounter[i]);
            }
        }
        entropy /= (double) data.numInstances(); //ini kayanya bisa diubah
        return entropy+Utils.log2(data.numInstances());
    }
    
    //Splits a dataset according to the values of a nominal attribute.
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                splitData[(int) inst.value(att)].add(inst);
            }
        return splitData;
    }
    
    //Computes information gain for an attribute.
    private double computeIG(Instances data, Attribute att) throws Exception {
        double IG = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
          if (splitData[j].numInstances() > 0) {
            IG -= ((double) splitData[j].numInstances() /
                         (double) data.numInstances()) *
              computeEntropy(splitData[j]);
          }
        }
        return IG;
    }
    
    public void buildClassifier(Instances i) throws Exception {
        //HAHAHA
    }
}
