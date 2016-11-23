/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.*;
import weka.core.Capabilities;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

// Reference: https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/clusterers/HierarchicalClusterer.java

/**
 *
 * @author vanyadeasy
 */
public class MyAgnes implements Clusterer {
    public static String SINGLE_LINKAGE = "single"; 
    public static String COMPLETE_LINKAGE = "complete"; 
    private DistanceFunction distanceFunction = null;
    private int numClusters = 2;
    private String linkType = COMPLETE_LINKAGE;
    private Instances instances;
    // Setiap ArrayList merepresentasikan iterasi
    // Setiap iterasi berisi sejumlah cluster
    // Setiap cluster berisi sejumlah ID instances
    private ArrayList<ArrayList<ArrayList<Integer>>> dendogram = new ArrayList<>();
    
    public MyAgnes(int numClusters, String type) {
        try {
            this.setNumClusters(numClusters);
            this.setLinkType(type);
        } catch (Exception ex) {
            Logger.getLogger(MyAgnes.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void setNumClusters(int num) throws Exception {
        if (num <= 0) {
            throw new Exception("Number of clusters should be > 0");
        }
        numClusters = num;
    }
    
    public void setLinkType(String linkType) throws Exception {
        if (!linkType.equals(SINGLE_LINKAGE) && !linkType.equals(COMPLETE_LINKAGE)) {
            throw new Exception("Wrong link type");
        }
        this.linkType = linkType;
    }
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        this.instances = instances;
        distanceFunction = new EuclideanDistance(instances);
        // Initiate all clusters (one cluster per instance)
        ArrayList<ArrayList<Integer>> firstIteration = new ArrayList<>();
        for(int i=0; i < instances.numInstances(); i++) {
            ArrayList<Integer> cluster = new ArrayList<>();
            cluster.add(i);
            firstIteration.add(cluster);
        }
        dendogram.add(firstIteration);
        
        // Print init clusters
        System.out.println("--------- INIT CLUSTERS ---------");
        for(int k=0;k<firstIteration.size();k++) System.out.println(k+"\t"+firstIteration.get(k));
            
        // If number of clusters = desired, terminate
        while(dendogram.get(dendogram.size()-1).size() > numClusters) {
            System.out.println("--------- #"+dendogram.size()+" ---------");
            ArrayList<ArrayList<Integer>> lastIteration = dendogram.get(dendogram.size()-1);
            double[][] distanceMatrix = countDistance(lastIteration);
            double minDistance = getMinimumDistance(distanceMatrix);
            
            System.out.println("Minimum distance: "+minDistance);
            
            ArrayList<ArrayList<Integer>> updated = new ArrayList<>(lastIteration);
            int i = 0;
            
            // Merge clusters based on distance matrix
            while(i < updated.size()) {
                int j = i+1;
                ArrayList<Integer> cluster = new ArrayList<>(updated.get(i));
                while(j < updated.size()) {
                    if(distanceMatrix[i][j] == minDistance) { // If distance = minimum distance, merge the cluster
                        cluster.addAll(updated.get(j));
                        updated.remove(updated.get(j));
                        distanceMatrix = countDistance(updated);
                        minDistance = getMinimumDistance(distanceMatrix);
                        break;
                    }
                    j++;
                }
                updated.set(i++, cluster);
            }
            dendogram.add(updated);
            
            // Print clusters
            for(int k=0;k<updated.size();k++) System.out.println(k+"\t"+updated.get(k));
        }
    }

    @Override
    public int clusterInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public int numberOfClusters() throws Exception {
        return numClusters;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        double[] p;
        if (numberOfClusters() == 0) {
            p = new double[1];
            p[0] = 1;
        } else {
            p = new double[numberOfClusters()];
            p[clusterInstance(instnc)] = 1.0; //@TODO : implemen clusterInstance
        }
        return p;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public double[][] countDistance(ArrayList<ArrayList<Integer>> clusters) {
        int nClusters = clusters.size();
        double[][] distanceMatrix = new double[nClusters][nClusters];
        
        for (int i = 0; i < nClusters; i++) {
            distanceMatrix[i][i] = Double.POSITIVE_INFINITY;
            for (int j = i + 1; j < nClusters; j++) {
                distanceMatrix[i][j] = distanceFunction.distance(instances.instance(clusters.get(i).get(0)), instances.instance(clusters.get(j).get(0)));
                for(int k = 0; k < clusters.get(i).size(); k++) {
                    for(int l = 0; l < clusters.get(j).size(); l++) {
                        double dist = distanceFunction.distance(instances.instance(clusters.get(i).get(k)), instances.instance(clusters.get(j).get(l)));
                        if(linkType.equals(SINGLE_LINKAGE)) { // If single, find minimum distance
                            if(dist < distanceMatrix[i][j]) distanceMatrix[i][j] = dist; 
                        }
                        else if(linkType.equals(COMPLETE_LINKAGE)) { // If complete, find maximum distance
                            if(dist > distanceMatrix[i][j]) distanceMatrix[i][j] = dist; 
                        }
                        else return null;
                    }
                }
                distanceMatrix[j][i] = Double.POSITIVE_INFINITY;
            }
        }
        return distanceMatrix;
    }
    
    public double getMinimumDistance(double[][] distanceMatrix) {
        return Arrays.stream(distanceMatrix)
                .flatMapToDouble(a -> Arrays.stream(a))
                .min()
                .getAsDouble();
    }
}
