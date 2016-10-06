/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.stream.DoubleStream;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author Venny
 */
public class MyC45 extends Classifier {
    private final double MISSING_VALUE = Double.NaN;
    private MyC45[] children; //node's successor
    private double label; //class value if node is leaf
    private Attribute splitAttr; //used for splitting
    private Attribute classAttr; //class attribute of dataset
    private double[] distribution; //class distribution for each label
    
    //returns default capabilities of the classifier
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.setMinimumNumberInstances(0);
        
        return result;
    }
    
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
    private Instances[] splitNominalData(Instances data, Attribute att) {
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
    
    // Splits a dataset according to a numeric attribute's threashold
    private Instances[] splitNumericData(Instances data, Attribute att, Double threshold) {
        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data, data.numInstances());
        splitData[1] = new Instances(data, data.numInstances());
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).value(att) <= threshold) {
                splitData[0].add(data.instance(i));
            } else {
                splitData[1].add(data.instance(i));
            }
        }
        splitData[0].compactify();
        splitData[1].compactify();
        return splitData;
    }
    
    // Computes information gain for a nominal attribute
    private double computeNominalIG(Instances data, Attribute att) throws Exception {
        double IG = computeEntropy(data);
        Instances[] splitData = splitNominalData(data, att);
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
    
    // Computes information gain for a numeric attribute
    private double computeNumericIG(Instances data, Attribute att, Double threshold) throws Exception {
        double IG = computeEntropy(data);
        Instances[] splitData = splitNumericData(data, att, threshold);
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
    
    
    // Calculate threshold for numeric attributes
    private double calculateThreshold(Instances data, Attribute att) throws Exception {
        data.sort(att);
        double threshold = data.instance(0).value(att);
        double IG = 0;
        for (int i = 0; i < data.numInstances()-1; i++){
            if (data.instance(i).classValue() != data.instance(i+1).classValue()) {
                double currentIG = computeNumericIG(data, att, data.instance(i).value(att));
                if (currentIG > IG) {
                    threshold = data.instance(i).value(att);
                }
            }
        }
        return threshold;
    }
    
    // Replace missing value with most common value of the attr among other examples with same target value 
    private void handleMissingValue (Instances data) {
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (data.instance(i).isMissing(j)) { // jika value untuk atribut ke-j missing
                    data.instance(i).setValue(data.attribute(j), mostCommonValue(data, data.attribute(j), data.instance(i).classValue()));
                }
            }
        }
    }
    
    private double mostCommonValue (Instances data, Attribute att, Double classValue) {
        List<Double> valList = Collections.list(att.enumerateValues());
        int [] attCount = new int [att.numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < att.numValues(); j++) {
                if (data.instance(i).value(att) == valList.get(j) && data.instance(i).classValue() == classValue) {
                    attCount[j]++; 
                }
            }
        }
        Arrays.sort(attCount);
        return valList.get(attCount[att.numValues() - 1]);
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
    private void buildTree(Instances data) throws Exception {
        //cek apakah terdapat instance yang dalam node ini
        if (data.numInstances()==0) {
            splitAttr = null;
            label = MISSING_VALUE;
            distribution = new double[data.numClasses()];
        } else {
            //jika ada, menghitung IG maksimum
            double[] infoGains = new double[data.numAttributes()];
            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                if (att.isNumeric()) {
                    double threshold = calculateThreshold(data, att);
                    infoGains[att.index()] = computeNumericIG(data, att, threshold);
                } else {
                    infoGains[att.index()] = computeNominalIG(data, att);
                }
            }
            //cek max IG
            int maxIG = maxIndex(infoGains);
            if (maxIG!=-1) { //kalo kosong
                splitAttr = data.attribute(maxIndex(infoGains));
            } else {
                Exception exception = new Exception("array null");
                throw exception;
            }

            //Membuat daun jika IG-nya 0
            if (Double.compare(infoGains[splitAttr.index()], 0) == 0) {
                splitAttr = null;
                distribution = new double[data.numClasses()];
                for (int i=0; i<data.numInstances(); ++i) {
                    Instance inst = (Instance) data.instance(i);
                    distribution[(int) inst.classValue()]++;
                }
                //normalisasi kelas distribusi
                double sum = DoubleStream.of(distribution).sum();
                if (!Double.isNaN(sum) && sum != 0) {
                    for (int i=0; i<distribution.length; ++i) {
                        distribution[i] /= sum;
                    }
                } else {
                    Exception exception = new Exception("Class distribution: NaN or sum=0");
                    throw exception;
                }
                label = maxIndex(distribution);
                classAttr = data.classAttribute();
            } else {
                // Membuat tree baru di bawah node ini
                Instances[] splitData;
                if (splitAttr.isNumeric()) {
                    double threshold = calculateThreshold(data, splitAttr);
                    splitData = splitNumericData(data, splitAttr, threshold);
                } else {
                    splitData = splitNominalData(data, splitAttr);
                }
                children = new MyC45[splitAttr.numValues()];
                for (int i=0; i<splitAttr.numValues(); i++) {
                    children[i] = new MyC45();
                    children[i].buildTree(splitData[i]);
                }
            }
        }
    }
    // builds J48 tree classifier
    @Override
    public void buildClassifier(Instances data) throws Exception{
        //cek apakah data dapat dibuat classifier
        getCapabilities().testWithFail(data);
        
        // Menghapus instances dengan missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        buildTree(data);
    }
    
    //classifies a given instance using the decision tree model
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (splitAttr == null) {
            return label;
        } else {
            return children[(int) instance.value(splitAttr)].classifyInstance(instance);
        }
    }
}
