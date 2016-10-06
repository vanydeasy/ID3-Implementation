/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
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
        System.out.print("filename (.arff or .csv): ");
        String filename = scan.nextLine();

        // Load data from ARFF or CSV
        Instances data = MyClassifier.loadData(filename);
        
        // Remove atribut
        List<Attribute> attr = Collections.list(data.enumerateAttributes());
        System.out.println("\nList of attributes\n-----------------");
        for(int i=0;i<attr.size();i++) {
            System.out.println(i+1 + ". " + attr.get(i).name());
        }
        System.out.print("Want to remove attribute (y/n)? ");
        if (scan.nextLine().equalsIgnoreCase("y")){
            System.out.print("Attribute to be removed: ");
            data = MyClassifier.removeAttribute(data, scan.nextLine());
        }
        
        // Build or Load model
        Classifier model;
        System.out.println("\nBuild or Load Model\n-----------------");
        System.out.println("1. Build Model");
        System.out.println("2. Load Existing Model");
        System.out.print("Choose: ");
        if (scan.nextLine().equals("1")){ // Build Model
            // Resample instances
            System.out.print("\nWant to resample (y/n)? ");
            if (scan.nextLine().equalsIgnoreCase("y")){
                System.out.print("Sample percentage (in %): ");
                data = MyClassifier.filterResample(data, Double.parseDouble(scan.nextLine()));
                System.out.print("Instances Resampled!\n");
            }
            
            // Choose classifier
            System.out.println("\nDecision Tree Classifiers\n-----------------");
            System.out.println("1. WEKA ID3");
            System.out.println("2. myID3");
            System.out.println("3. WEKA J48");
            System.out.println("4. myC45");
            System.out.print("Choose classifier: ");
            
            String input = scan.nextLine();
            switch (input) {
                case "2":
                    model = new MyID3();
                    break;
                case "3":
                    model = new J48();
                    break;
                case "4":
                    model = new MyC45();
                    break;
                default:
                    model = new Id3();
                    break;
            }
            
            if (input.equals("2") || input.equals("4")) { //MyID3 or MyC45
                Instances unlabeled = ConverterUtils.DataSource.read(filename);
                unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
                Instances labeled = new Instances(unlabeled);

                // label instances
                for (int i=0; i<unlabeled.numInstances(); ++i) {
                    double clsLabel = model.classifyInstance(unlabeled.instance(i));
                    labeled.instance(i).setClassValue(clsLabel);
                    // System.out.println(labeled.instance(i));
                } 
            }
            // 10-fold cross validation or Percentage split
            System.out.println("\nEvaluation Method\n-----------------");
            System.out.println("1. 10-fold cross validation");
            System.out.println("2. Percentage split");
            System.out.print("Choose evaluation method: ");
            if (scan.nextLine().equals("1")){
                // Build Classifier
                model.buildClassifier(data);
                System.out.println(model.toString());
                
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(model, data, 10, new Random());
                System.out.println(eval.toSummaryString("\n10-Fold Cross Validation\n============", false));
            } else {
                System.out.print("Training data percentage (in %): ");
                int trainSize = (int) Math.round(data.numInstances() * Double.parseDouble(scan.nextLine())/100);
                int testSize = data.numInstances() - trainSize;
                Instances train = new Instances(data, 0, trainSize);
                Instances test = new Instances(data, trainSize, testSize);
                
                // Build Classifier
                model.buildClassifier(train);
                System.out.println(model.toString());
                
                Evaluation eval = new Evaluation(test);
                eval.evaluateModel(model, test);
                System.out.println(eval.toSummaryString("\nPercentage Split Validation\n============", false));
            }
            
            // Save Model
            System.out.print("Want to save model (y/n)? ");
            if (scan.nextLine().equalsIgnoreCase("y")){
                System.out.print("filename: ");
                MyClassifier.SaveModel(model, scan.next());
                System.out.print("Model Saved!\n");
            }
            
        } else { // Load Model 
            System.out.print("\nLoad Model\n----------\nfilename (.model): ");
            model = MyClassifier.LoadModel(scan.next());
            System.out.print("Model Loaded!\n");
        }
        
        // Test with given test set
        System.out.print("Want to test model using a new test set (y/n)? ");
        if (scan.nextLine().equalsIgnoreCase("y")){
            System.out.print("filename: ");
            Instances testSet = MyClassifier.loadData(scan.nextLine());
            Evaluation eval = new Evaluation(testSet);
            eval.evaluateModel(model, testSet);
            System.out.println(eval.toSummaryString("\nTest Set Result\n============", false));
        }
        
        // Prediction using user input
        System.out.print("Want to predict an unseen instance (y/n)? ");
        if (scan.nextLine().equalsIgnoreCase("y")){
            System.out.println("\nClassify An Unseen Instance\n-------------------------");
            List<Attribute> attrNew = Collections.list(data.enumerateAttributes());
            Instance predInst = new Instance(attrNew.size());
            for(int i=0;i<attrNew.size();i++) {
                System.out.print("Data "+attrNew.get(i).name()+": ");
                if(attrNew.get(i).isNumeric())
                    predInst.setValue(attrNew.get(i),scan.nextDouble());
                else
                    predInst.setValue(attrNew.get(i),scan.next());
            }
            predInst.setDataset(data);
            String prediction = data.classAttribute().value((int)model.classifyInstance(predInst));
            System.out.println("The predicted value of instance is "+prediction);
        }
    }
    
    public static Instances loadData(String filename) throws FileNotFoundException, IOException {
        Instances data;
        if (filename.substring(filename.lastIndexOf(".") + 1).equals("arff")){
            BufferedReader br = new BufferedReader(new FileReader(filename));
            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(br);
            data = arff.getData();
        } else {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filename));
            data = loader.getDataSet();
        }
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    public static Instances removeAttribute (Instances data, String attr) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices(attr);
        remove.setInvertSelection(false);
        remove.setInputFormat(data);
        
        return Filter.useFilter(data, remove);
    }
    
    public static void SaveModel(Classifier model, String filename) throws Exception {
        weka.core.SerializationHelper.write(filename, model);
    }
    
    public static Classifier LoadModel(String filename) throws Exception {
        Classifier cls = (Classifier) weka.core.SerializationHelper.read(filename);
        return cls;
    }
    
    public static Instances filterResample(Instances data, double sampleSize) {
	final Resample filter = new Resample();
	Instances filteredIns = null;
	filter.setBiasToUniformClass(1.0);
	try {
		filter.setInputFormat(data);
		filter.setNoReplacement(false);
		filter.setSampleSizePercent(sampleSize);
		filteredIns = Filter.useFilter(data, filter);
	} catch (Exception e) {
		e.printStackTrace();
	}
	return filteredIns;
    }
}
