/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
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
    private double threshold;
    
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
        // OPSI 1
        // Sort berdasarkan nilai atribut, tiap batas pergantian kelas di split dan dihitung IGnya
        // Dari semua kemungkinan tempat split, ambil yang IGnya paling besar
        /* data.sort(att);
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
        return threshold; */
        
        // OPSI 2
        // threshold = min+max/2
        /* double min = data.instance(0).value(att);
        double max = data.instance(0).value(att);
        for (int i=1; i< data.numInstances(); i++) {
            if (data.instance(i).value(att) < min) min = data.instance(i).value(att);
            if (data.instance(i).value(att) > max) max = data.instance(i).value(att);
        }
        return min+max/2; */
        
        // OPSI 3
        // threshold = avg
        double sum = 0;
        for (int i=1; i< data.numInstances(); i++) {
            sum += data.instance(i).value(att);
        }
        return sum/data.numInstances();
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
        if (att.isNumeric()) {
            double sum = 0;
            for (int i=1; i< data.numInstances(); i++) {
                sum += data.instance(i).value(att);
            }
            return sum/data.numInstances();
        } else {
            List<String> valList = Collections.list(att.enumerateValues());
            int [] attCount = new int [att.numValues()];
            for (int i = 0; i < data.numInstances(); i++) {
                for (int j = 0; j < att.numValues(); j++) {
                    if (!data.instance(i).isMissing(att)) {
                        if (data.instance(i).stringValue(att).equals(valList.get(j)) && data.instance(i).classValue() == classValue) {
                            attCount[j]++; 
                        }
                    }
                }
            }
            int maxIndex = 0;
            int max = attCount[0];
            for (int j = 1; j < attCount.length; j++){
               if (attCount[j] > max) maxIndex = j;
            }
            System.out.println(valList.get(maxIndex));
            return att.indexOfValue(valList.get(maxIndex));
        }
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
        handleMissingValue(data);
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
                    threshold = calculateThreshold(data, att);
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
                    splitData = splitNumericData(data, splitAttr, threshold);
                    children = new MyC45[2];
                    for (int i=0; i<2; i++) {
                        children[i] = new MyC45();
                        children[i].buildTree(splitData[i]);
                    }
                } else {
                    splitData = splitNominalData(data, splitAttr);
                    children = new MyC45[splitAttr.numValues()];
                    for (int i=0; i<splitAttr.numValues(); i++) {
                        children[i] = new MyC45();
                        children[i].buildTree(splitData[i]);
                    }
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
        
        System.out.println("_______________");
        System.out.println(this.toString());
        System.out.println("_______________");
        
        //pruning(this,this,data);
    }
    
    //classifies a given instance using the decision tree model
    @Override
    public double classifyInstance(Instance instance){
        if (splitAttr == null) {
            return label;
        } else {
            if (splitAttr.isNumeric()){
                if (instance.value(splitAttr) <= threshold) {
                    return children[0].classifyInstance(instance);
                }
                return children[1].classifyInstance(instance);
            }
            return children[(int) instance.value(splitAttr)].classifyInstance(instance);
        }
    }
    
    // Prints the decision tree using the private toString method from below.
    @Override
    public String toString() {
        if ((distribution == null) && (children == null)) {
            return "\nMyC45: No DT model";
        }
        return "\nMyC45\n" + toString(0);
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
            if (splitAttr.isNumeric()) {
                result.append("\n");
                int j=0;
                while(j<level) {
                    result.append("|  ");
                    j++;
                }
                result.append(splitAttr.name()).append(" <= ").append(threshold);
                result.append(children[0].toString(level+1));
                
                result.append("\n");
                j=0;
                while(j<level) {
                    result.append("|  ");
                    j++;
                }
                result.append(splitAttr.name()).append(" > ").append(threshold);
                result.append(children[1].toString(level+1));
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
        }
        return result.toString();
    }
    
    public void pruning(MyC45 root, MyC45 node, Instances test) {
        if(node.splitAttr == null) { // LEAF
            
        }
        else {
            for(int i=0;i<node.children.length;i++) {
                if(node.children[i].splitAttr != null) { // If child not leaf
                    System.out.println("ORIGINAL");
                    System.out.println(root.toString());
                    
                    Double errorBeforePruning = 0.00;
                    Double errorAfterPruning = 0.00;
                    try {
                        // Calculating error 
                        errorBeforePruning = calculateError(test)*1.00;
                    } catch (Exception ex) {
                        Logger.getLogger(MyC45.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    Attribute tempSplitAttr = node.children[i].splitAttr;
                    Double tempLabel = node.children[i].label;
                    
                    node.children[i].splitAttr = null;
                    // TODO: Label to pruned
                    node.children[i].label = MISSING_VALUE;
                    
                    try {
                        // Calculating error
                        errorAfterPruning = calculateError(test)*1.00;
                    } catch (Exception ex) {
                        Logger.getLogger(MyC45.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    System.out.println("PRUNED");
                    System.out.println(root.toString());

                    if(errorBeforePruning < errorAfterPruning) {
                        node.children[i].splitAttr = tempSplitAttr;
                        node.children[i].label = tempLabel;
                    }
                }
                pruning(root, node.children[i], test);
            }
        }
    }
    
    public int calculateError(Instances test) {
        int incorrect = 0;
        for(int i=0;i<test.numInstances();i++) {
            if(classifyInstance(test.instance(i)) != test.instance(i).classValue()) incorrect++;
        }
        return incorrect/test.numInstances();
    }
    
    public Instances filterByAttributesValue(Instances instances, Attribute attr, Double label) {
        Instances filtered = new Instances(instances);
        for(int i=0;i<instances.numInstances();i++) {
            if(instances.instance(i).value(attr) != label) {
                filtered.delete(i);
            }
        }
        return filtered;
    }
}
