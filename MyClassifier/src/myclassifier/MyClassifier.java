/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Venny
 */
public class MyClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        Scanner scan = new Scanner(System.in);
        
        // Load data from ARFF or CSV
        Instances data;
        if (args[0].substring(args[0].lastIndexOf(".") + 1).equals("arff")){
            BufferedReader br = new BufferedReader(new FileReader(args[0]));
            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(br);
            data = arff.getData();
        } else {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(args[0]));
            data = loader.getDataSet();
        }
        data.setClassIndex(data.numAttributes() - 1);
        
        // Remove atribut
        Remove remove = new Remove();
        List<Attribute> attr = Collections.list(data.enumerateAttributes());
        Instance predInst = new Instance(attr.size());
        
        System.out.println("List of attributes\n-----------------");
        for(int i=0;i<attr.size();i++) {
            System.out.println(i+1 + ". " + attr.get(i).name());
        }
        
        System.out.print("Attribute to be removed: ");
        remove.setAttributeIndices(scan.next());
        remove.setInvertSelection(false);
        remove.setInputFormat(data);
        
        Instances dataNew = Filter.useFilter(data, remove);
        
        // Build Naive Bayes Model
        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(dataNew);
        System.out.println(model.toString());
        
        // Save Model
        weka.core.SerializationHelper.write(args[0].substring(0, args[0].indexOf('.')) + "_naivebayes.model", model);
        
        // Load Model
        Classifier cls = (Classifier) weka.core.SerializationHelper.read(args[0].substring(0, args[0].indexOf('.')) + "_naivebayes.model");
        
        // 10-fold Cross Validation Evaluation
        Evaluation eval = new Evaluation(dataNew);
        eval.crossValidateModel(cls, data, 10, new Random());
        System.out.println(eval.toSummaryString("\n\n\n\nNaive Bayes 10-Fold Cross Validation\n============\n", false));
        
        // Prediction using user input
        for(int i=0;i<attr.size();i++) {
            System.out.print("Data "+attr.get(i).name()+": ");
            if(attr.get(i).isNumeric())
                predInst.setValue(attr.get(i),scan.nextDouble());
            else
                predInst.setValue(attr.get(i),scan.next());
        }
        predInst.setDataset(data);
        String prediction = data.classAttribute().value((int)cls.classifyInstance(predInst));
        System.out.println("The predicted value of instance is "+prediction);
    }
}
