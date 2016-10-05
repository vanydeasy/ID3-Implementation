/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.core.Attribute;
import java.util.Enumeration;
import java.util.stream.DoubleStream;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author Pipin
 */
public class MyID3 extends Classifier {
    private final double MISSING_VALUE = Double.NaN;
    private MyID3[] children; //node's successor
    private double label; //class value if node is leaf
    private Attribute splitAttr; //used for splitting
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
        for (int j=0; j<att.numValues(); j++) {
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
            if (splitdata.numInstances()>0) {
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
    
    // Returns default capabilities of the classifier.
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.setMinimumNumberInstances(0);
        return result;
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
                infoGains[att.index()] = computeIG(data, att);
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
                Instances[] splitData = splitData(data, splitAttr);
                children = new MyID3[splitAttr.numValues()];
                for (int i=0; i<splitAttr.numValues(); i++) {
                    children[i] = new MyID3();
                    children[i].buildTree(splitData[i]);
                }
            }
        }
    }
    
    //build ID3 classifier
    @Override
    public void buildClassifier(Instances inst) throws Exception {
        //cek apakah data dapat dibuat classifier
        getCapabilities().testWithFail(inst);
        
        // Menghapus instances dengan missing class
        inst = new Instances(inst);
        inst.deleteWithMissingClass();
        buildTree(inst);
    }
    
    //classifies a given instance using the decision tree model
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyID3 cannot handle missing values");
        }
        if (splitAttr == null) {
            return label;
        } else {
            return children[(int) instance.value(splitAttr)].classifyInstance(instance);
        }
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
    public String toString(int level) {
        StringBuilder result = new StringBuilder();
        if (splitAttr == null) {
            if (Instance.isMissingValue(label)) {
                result.append(": null");
            } else {
                result.append(": ").append(classAttr.value((int) label));
            }
        } else {
            for (int i=0; i<splitAttr.numValues(); i++) {
                result.append("\n");
                
                int j=0;
                while(j<level) {
                    result.append("|  ");
                    j++;
                }
                result.append(splitAttr.name()).append(" = ").append(splitAttr.value(i));
                result.append(children[i].toString(level+1));
            }
        }
        return result.toString();
    }
}