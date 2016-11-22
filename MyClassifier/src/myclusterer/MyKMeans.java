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
    private int maxIterations = 500;
    private Instances centroids = null;
    
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
        // INITIALIZATION:
        boolean convergence = false;
        int numIterations = 1;
        centroids = new Instances(instances, 3);
        Instances[] clusters = new Instances[numClusters];
        
        for (int i = 0; i < numClusters; i++) {
            clusters[i] = new Instances(instances, instances.numInstances());
            
            // Pick centroids randomly
            Random rand = new Random();
            int randomNum = rand.nextInt(instances.numAttributes());
            centroids.add(instances.instance(randomNum));
        }
        
        while (convergence == false && numIterations < maxIterations) {
            // ASSIGNMENT:
            for (int i = 0; i < instances.numInstances(); i++) {
                int clusterNo = clusterInstance(instances.instance(i));
                clusters[clusterNo].add(instances.instance(i));
            }
            
            for (int i = 0; i < numClusters; i++) {
                clusters[i].compactify();
            }

            // UPDATE CENTROID:
            Instances oldCentroids = centroids;
            for (int i = 0; i < numClusters; i++) { // for each cluster
                for (int j = 0; j < clusters[i].numAttributes(); j++) { // for each attribute
                    
                    // if attribute is numeric calculate mean
                    if (clusters[i].attribute(j).isNumeric()) {
                        double avg = clusters[i].attributeStats(j).numericStats.mean;
                        centroids.instance(i).setValue(j, avg);
                    
                    // if attribute is nominal find modes
                    } else if (clusters[i].attribute(j).isNominal()) {
                        int[] values = clusters[i].attributeStats(j).nominalCounts;
                        int max = 0;
                        int idxMax = 0;
                        for (int k = 0; k < clusters[i].attribute(j).numValues(); k++) {
                            if (max > values[k]) {
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
                for (int j = 0; j < clusters[i].numAttributes(); j++) {
                    if (centroids.instance(i).value(j) != oldCentroids.instance(i).value(j)) {
                        conv = false;
                        break;
                    }
                }
                if (!conv) break;
            }
            convergence = conv;
            
            numIterations++;
        }
            
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        int centroidIdx = 0;
        double distance = 0;
        for (int i = 0; i < centroids.numInstances(); i++) {
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
}
