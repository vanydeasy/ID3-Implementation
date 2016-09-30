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
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

/**
 *
 * @author Venny
 */
public class MyClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
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
        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(data);
        System.out.println(model.toString());
        
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random());
        System.out.println(eval.toSummaryString("\n\n\n\nNaive Bayes 10-Fold Cross Validation\n============\n", false));
    }
    
}
