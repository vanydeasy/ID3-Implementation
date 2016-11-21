/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

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
    private int numClusters = 2;
    private int maxIterations = 500;
    private Instances centroids = null;
    private int[] assignment = null;
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        // INITIALIZATION:
        boolean convergence = false;
        int numIterations = 1;
        centroids = new Instances(instances, 3);
        assignment = new int[instances.numInstances()];
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
                assignment[i] = clusterInstance(instances.instance(i));
                clusters[assignment[i]].add(instances.instance(i));
            }

            // UPDATE CENTROID:
            for (int i = 0; i < numClusters; i++) { // for each cluster
                double[] attrAverage = new double[clusters[i].numAttributes()];
                for (int j = 0; j < clusters[i].numAttributes(); j++) { // for each attribute
                    if (clusters[i].attribute(j).isNumeric()) { // if attribute is numeric calculate mean
                        attrAverage[j] = clusters[i].attributeStats(j).numericStats.mean;
                    } else if (clusters[i].attribute(j).isNominal()) { // if attribute is nominal find modes
                        int[] values = clusters[i].attributeStats(j).nominalCounts;
                        // cari index di mana value max
                        // attrAverage = index tersebut
                    }
                }
            }   
            // If old centroids = new centroids
                // convergence = true;
            
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
