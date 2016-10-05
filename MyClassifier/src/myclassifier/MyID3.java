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
    private final double MISSING_VALUE = Double.NaN;
    private Id3[] children; //node's successor
    private double label; //class value if node is leaf
    private Attribute attribute; //used for splitting
    private Attribute classAttr; //class attribute of dataset
    private double[] distribution; //class distribution for each label
    
     //Computes the entropy of a dataset.
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
    
    //Splits a dataset according to the values of a nominal attribute.
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
    
    //Computes information gain for an attribute.
    private double computeIG(Instances data, Attribute att) throws Exception {
        double IG = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double dataNumInstances = data.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                IG -= proportion * computeEntropy(splitdata);
            }
        }
        return IG;
    }
    
    //return the index with largest value from array
    private int maxIndex(double[] array) {
        double max=0;
        int index=0;
        if (array.length>0) {
            for (int i=0; i<array.length; ++i) {
                if (array[i]>max) {
                    max=array[i];
                    index=i;
                }
            }
            return index;
        } else {
            return -1;
        }
    }
    
    // Creates an Id3 tree.
    private void builTree(Instances data) throws Exception {
        //cek apakah terdapat instance yang dalam node ini
        if (data.numInstances() == 0) {
            attribute = null;
            label = MISSING_VALUE;
            distribution = new double[data.numClasses()];
        } else {
            //jika ada, menghitung IG maksimum
            double[] infoGains = new double[data.numAttributes()];
            
            //@TODO : UBAH SEMUA INSTANCE KE NOMINAL
            
            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                infoGains[att.index()] = computeIG(data, att);
            }

            attribute = data.attribute(maxIndex(infoGains));

            //@TODO : CEK IG. JIKA NOL, BUAT DAUN
            //JIKA TIDAK, BUAT TREE BARU (REKURSIF)
        }
    }
    
    public void buildClassifier(Instances i) throws Exception {
        //HAHAHA
    }
    
    // Prints the decision tree using the private toString method from below.
    @Override
    public String toString() {

        if ((distribution == null) && (children == null)) {
            return "MyID3: No DT model";
        }
        return "MyID3\n\n" + toString(0);
    }
    
    //Outputs a tree at a certain level
    private String toString(int level) {
        StringBuilder result = new StringBuilder();
        if (attribute == null) {
            if (Instance.isMissingValue(label)) {
                result.append(": null");
            } else {
                result.append(": ").append(attribute.value((int) label));
            }
        } else {
            for (int i=0; i<attribute.numValues(); i++) {
                result.append("\n");
                
                int j=0;
                while(j<level) {
                    result.append("|  ");
                    j++;
                }
                result.append(attribute.name()).append(" = ").append(attribute.value(i));
                //result.append(children[i].toString(level+1)); //WHYYYYYY
            }
        }
        return result.toString();
    }
}
