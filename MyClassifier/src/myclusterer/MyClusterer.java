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
import weka.core.SelectedTag;
import static weka.clusterers.HierarchicalClusterer.TAGS_LINK_TYPE;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

/**
 *
 * @author vanyadeasy
 */
public class MyClusterer {
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
                System.out.print("\nNumber of clusters: ");
                SimpleKMeans kmeans = new SimpleKMeans();
                kmeans.setSeed(3);
                kmeans.setPreserveInstancesOrder(true);
                kmeans.setNumClusters(Integer.parseInt(scan.nextLine()));
                kmeans.buildClusterer(data);
                eval.setClusterer(kmeans);
                eval.evaluateClusterer(data);
                break;
            case "2":
                System.out.print("\nNumber of clusters: ");
                MyKMeans mykmeans = new MyKMeans(Integer.parseInt(scan.nextLine()));
                mykmeans.buildClusterer(data);
                eval.setClusterer(mykmeans);
                eval.evaluateClusterer(data);
                break;
            case "3":
                HierarchicalClusterer agnes = new HierarchicalClusterer();
                agnes.setLinkType(new SelectedTag(1, TAGS_LINK_TYPE));
                agnes.buildClusterer(data);
                eval.setClusterer(agnes);
                eval.evaluateClusterer(data);
                break;
            case "4":
                System.out.print("\nNumber of clusters: ");
                int numClusters = Integer.parseInt(scan.nextLine());
                System.out.print("\nChoose Type (single/complete)? ");
                String type = scan.nextLine().equals("single") ? MyAgnes.SINGLE_LINKAGE : MyAgnes.COMPLETE_LINKAGE;
                MyAgnes myAgnes = new MyAgnes(numClusters, MyAgnes.COMPLETE_LINKAGE);
                myAgnes.buildClusterer(data);
                eval.setClusterer(myAgnes);
                eval.evaluateClusterer(data);
                break;
            default:
                System.out.println("Option not found!");
                break;
        }

        System.out.println("\nCluster Evaluation:\n "+eval.clusterResultsToString());
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
