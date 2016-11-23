/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import myclassifier.MyClassifier;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

/**
 *
 * @author vanyadeasy
 */
public class MyClusterer {
    private static String SINGLE_LINKAGE = "single"; 
    private static String COMPLETE_LINKAGE = "complete"; 
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        // Load data from ARFF or CSV
        Scanner scan = new Scanner(System.in);
        System.out.print("filename (.arff or .csv): ");
        String filename = scan.nextLine();
        System.out.print("Load class attribute (Y/N)? ");
        Instances data;
        if (scan.nextLine().equalsIgnoreCase("y")){
            data = MyClusterer.loadData(filename, true);
        } else {
            data = MyClusterer.loadData(filename, false);
        }
        
        ClusterEvaluation eval = new ClusterEvaluation();
        
        // Choose classifier
        System.out.println("\nClustering Algorithm\n-----------------");
        System.out.println("1. WEKA KMeans");
        System.out.println("2. myKMeans");
        System.out.println("3. WEKA Agnes");
        System.out.println("4. myAgnes");
        System.out.print("Choose clusterer: ");

        String input = scan.nextLine();
        switch (input) {
            case "1":
                SimpleKMeans kmeans = new SimpleKMeans();
                kmeans.setSeed(3);
                kmeans.setPreserveInstancesOrder(true);
                kmeans.setNumClusters(2);
                kmeans.buildClusterer(data);
                eval.setClusterer(kmeans);
                eval.evaluateClusterer(data);
                break;
            case "2":
                MyKMeans mykmeans = new MyKMeans(3);
                mykmeans.buildClusterer(data);
                break;
            case "3":
                HierarchicalClusterer agnes = new HierarchicalClusterer();
                break;
            case "4":
                MyAgnes myAgnes = new MyAgnes(2, SINGLE_LINKAGE);
                myAgnes.buildClusterer(data);
                eval.setClusterer(myAgnes);
                break;
            default:
                System.out.println("Option not found!");
                break;
        }

        System.out.println("Cluster Evaluation: "+eval.clusterResultsToString());
    }
    
    public static Instances loadData(String filename, boolean loadClass) throws FileNotFoundException, IOException {
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
        if (loadClass) data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
}
