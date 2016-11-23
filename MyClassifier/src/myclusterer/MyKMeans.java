/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

import java.util.ArrayList;
import java.util.Random;
import weka.clusterers.*;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Venny
 */
public class MyKMeans implements Clusterer, CapabilitiesHandler {
    private DistanceFunction distanceFunction = new EuclideanDistance();
    private int numClusters;
    private int maxIterations = 100;
    private int numIterations;
    private Instances centroids = null;
    private Instances[] clusters;
    
    public MyKMeans(int numClusters) {
        this.numClusters = numClusters;
    }
    
    private ArrayList<Integer> initializeCentroids(Instances data) {
        Random random = new Random();
        ArrayList<Integer> centroids = new ArrayList<>();
        while (centroids.size() < numClusters) {
            Integer next;
            do {
                next = random.nextInt(data.numInstances());
            } while (centroids.contains(next));
            centroids.add(next);
        }
        return centroids;
    }
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        
        // INITIALIZATION:
        boolean convergence = false;
        numIterations = 1;
        centroids = new Instances(instances, 3);
        clusters = new Instances[numClusters];
        ArrayList<Integer> center = initializeCentroids(instances);
        distanceFunction = new EuclideanDistance(instances);
        
        System.out.println("\nCENTROID INITIALIZATION");
        for (int i = 0; i < numClusters; i++) {
            clusters[i] = new Instances(instances, instances.numInstances());
            centroids.add(instances.instance(center.get(i)));
            
            System.out.print(i+ ": ");
            for (int j = 0; j < centroids.instance(i).numAttributes(); j++) {
                if (centroids.instance(i).attribute(j).isNominal()) {
                    System.out.print(centroids.instance(i).stringValue(j) + ", ");
                } else {
                    System.out.print(centroids.instance(i).value(j) + ", ");
                }
            }
            System.out.println("");
        }
        System.out.println("");
        Instances oldCentroids = new Instances(instances, 3);
        for (int i = 0; i < numClusters; i++) {
            oldCentroids.add(centroids.instance(i));
        }
        
        while (convergence == false && numIterations <= maxIterations) {
            // Empty each cluster
            for (int i = 0; i < numClusters; i++) {
                clusters[i].delete();
            }
            
            // ASSIGNMENT:
            for (int i = 0; i < instances.numInstances(); i++) {
                int clusterNo = clusterInstance(instances.instance(i));
                clusters[clusterNo].add(instances.instance(i));
            }
            
            for (int i = 0; i < numClusters; i++) {
                clusters[i].compactify();
            }

            // UPDATE CENTROID:
            for (int i = 0; i < numClusters; i++) { // for each cluster
                for (int j = 0; j < clusters[i].numAttributes(); j++) { // for each attribute
                    
                    // if attribute is numeric calculate mean
                    if (clusters[i].attribute(j).isNumeric()) {
                        double avg = clusters[i].attributeStats(j).numericStats.mean;
                        double roundedAvg = (double) Math.round(avg * 1000000) / 1000000;
                        centroids.instance(i).setValue(j, avg);
                    
                    // if attribute is nominal find modes
                    } else if (clusters[i].attribute(j).isNominal()) {
                        int[] values = clusters[i].attributeStats(j).nominalCounts;
                        int max = 0;
                        int idxMax = 0;
                        for (int k = 0; k < clusters[i].attribute(j).numValues(); k++) {
                            if (max < values[k]) {
                                max = values[k];
                                idxMax = k;
                            }
                        }
                        centroids.instance(i).setValue(j, clusters[i].attribute(j).value(idxMax));
                    }
                }            
            }   
            
            // DETERMINE CONVERGENCE:
            boolean conv = true;
            for (int i = 0; i < numClusters; i++) {
                for (int j = 0; j < centroids.numAttributes(); j++) {
                    if (centroids.instance(i).value(j) != oldCentroids.instance(i).value(j)) {
                        conv = false;
                        break;
                    }
                }
                if (!conv) break;
            }
            convergence = conv;
            if (!convergence) {
                numIterations++;
                for (int i = 0; i < numClusters; i++) {
                    for (int j = 0; j < centroids.numAttributes(); j++) {
                        oldCentroids.instance(i).setValue(j, centroids.instance(i).value(j));
                    }
                }
            }
        }
            
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        int centroidIdx = 0;
        double distance = Integer.MAX_VALUE;
        for (int i = 0; i < numClusters ; i++) {
            double currentDistance = distanceFunction.distance(centroids.instance(i), instance);
            if (currentDistance < distance) {
                distance = currentDistance;
                centroidIdx = i;
            }
        }
        return centroidIdx;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int numberOfClusters() throws Exception {
        return this.numClusters;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NO_CLASS);
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        
        return result;
    }
    
    public int getNumClusters() {
        return this.numClusters;
    }
    
    public void setNumClusters(int numClusters) throws Exception {
        if (numClusters <= 0) {
            throw new Exception("Number of clusters must be > 0");
        }
        this.numClusters = numClusters;
    }
    
    public int getMaxIterations() {
        return this.maxIterations;
    }
     
    public void setMaxIterations(int max) throws Exception {
        if (max <= 0) {
            throw new Exception("Maximum number of iterations must be > 0");
        }
        this.maxIterations = max;
    }
    
    public DistanceFunction getDistanceFunction() {
        return this.distanceFunction;
    }
    
    public void setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }
    
    public Instances getClusterCentroids() {
        return this.centroids;
    }
    
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("\nMyKMeans\n========\n\n");
        result.append("Number of clusters: "+ numClusters + "\n");
        result.append("Number of iterations: " + numIterations + "\n\n");
        
        for (int i = 0; i < numClusters; i++) {
            result.append(" ----- CLUSTER #"+ i + " CENTROID -----\n");
            for (int j = 0; j < clusters[i].numAttributes(); j++) {
                result.append(clusters[i].attribute(j).name() + " = ");
                if (clusters[i].attribute(j).isNominal()) {
                    result.append(centroids.instance(i).stringValue(j) + "\n");
                } else {
                    result.append(centroids.instance(i).value(j) + "\n");
                }
            }
            result.append("\n");
        }
        
        return result.toString();
    }
}
