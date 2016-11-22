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
    private static String SINGLE_LINKAGE = "single"; 
    private static String COMPLETE_LINKAGE = "complete"; 
    private DistanceFunction distanceFunction = null;
    private int numClusters = 2;
    private String linkType = COMPLETE_LINKAGE;
    private ArrayList<ArrayList<ArrayList<Instance>>> dendogram = new ArrayList<>();
    
    public MyAgnes(int numClusters, String type) {
        try {
            this.numClusters = numClusters;
            this.setLinkType(type);
        } catch (Exception ex) {
            Logger.getLogger(MyAgnes.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        distanceFunction = new EuclideanDistance(instances);
        // Initiate all clusters (one cluster per instance)
        ArrayList<ArrayList<Instance>> firstIteration = new ArrayList<>();
        for(int i=0; i < instances.numInstances(); i++) {
            ArrayList<Instance> cluster = new ArrayList<>();
            cluster.add(instances.instance(i));
            firstIteration.add(cluster);
        }
        dendogram.add(firstIteration);
        
        while(dendogram.get(dendogram.size()-1).size() > numClusters) {
            System.out.println("--------- #"+dendogram.size()+" ---------");
            ArrayList<ArrayList<Instance>> lastIteration = dendogram.get(dendogram.size()-1);
            double[][] distances = countDistance(lastIteration);
            double minDistance = Arrays.stream(distances)
                .flatMapToDouble(a -> Arrays.stream(a))
                .min()
                .getAsDouble();
            
            System.out.println("Minimum distance: "+minDistance);
            ArrayList<ArrayList<Instance>> updated = new ArrayList<>(lastIteration);
            
            int i = 0;
            while(i < updated.size()) {
                int j = i+1;
                ArrayList<Instance> cluster = new ArrayList<>(updated.get(i));
                while(j < updated.size()) {
                    System.out.println(">> "+i+","+j+"\t"+updated.get(i)+" "+updated.get(j));
                    if(distances[i][j] == minDistance) {
                        System.out.println(">>>> Merged");
                        cluster.addAll(updated.get(j));
                        updated.remove(updated.get(j));
                        distances = countDistance(updated);
                        minDistance = Arrays.stream(distances)
                            .flatMapToDouble(a -> Arrays.stream(a))
                            .min()
                            .getAsDouble();
                        break;
                    }
                    j++;
                }
                updated.set(i++, cluster);
            }
            dendogram.add(updated);
            
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

    public void setNumClusters(int num) throws Exception {
        if (num <= 0) {
            throw new Exception("Number of clusters should be > 0");
        }
        numClusters = num;
    }
    
    public void setLinkType(String linkType) throws Exception {
        if (linkType != SINGLE_LINKAGE || linkType != COMPLETE_LINKAGE) {
            throw new Exception("Wrong link type");
        }
        this.linkType = linkType;
    }
    
    public double[][] countDistance(ArrayList<ArrayList<Instance>> instances) {
        int nClusters = instances.size();
        double[][] distanceMatrix = null;
        distanceMatrix = new double[nClusters][nClusters];
        
        for (int i = 0; i < nClusters; i++) {
            distanceMatrix[i][i] = Double.POSITIVE_INFINITY;
            for (int j = i + 1; j < nClusters; j++) {
                distanceMatrix[i][j] = distanceFunction.distance(instances.get(i).get(0), instances.get(j).get(0));
                for(int k = 0; k < instances.get(i).size(); k++) {
                    for(int l = 0; l < instances.get(j).size(); l++) {
                        double dist = distanceFunction.distance(instances.get(i).get(k), instances.get(j).get(l));
                        if(linkType.equals(SINGLE_LINKAGE)) {
                            if(dist < distanceMatrix[i][j]) distanceMatrix[i][j] = dist; 
                        }
                        else if(linkType.equals(COMPLETE_LINKAGE)) {
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
    
}
