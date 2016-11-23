/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

import java.util.ArrayList;
import java.util.Random;
import weka.clusterers.*;
import weka.core.Attribute;
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
    
    private int fullNumInstances;
    private Instance fullAttrAvg;
    
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
        
        fullAttrAvg = instances.instance(0);
        fullNumInstances = instances.numInstances();
        double[] attrAvg = calculateMean(instances);
        for (int i = 0; i < instances.numAttributes(); i++) {
            fullAttrAvg.setValue(i, attrAvg[i]);
        }
        
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
                double[] mean = calculateMean(clusters[i]);
                for (int j = 0; j < clusters[i].numAttributes(); j++) { // for each attribute
                    if (clusters[i].attribute(j).isNumeric()) {
                        centroids.instance(i).setValue(j, mean[j]);
                    } else if (clusters[i].attribute(j).isNominal()) {
                        centroids.instance(i).setValue(j, clusters[i].attribute(j).value((int)mean[j]));
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
    
    public double[] calculateMean(Instances data) {
        double[] avg = new double[data.numAttributes()];
        for (int j = 0; j < data.numAttributes(); j++) { // for each attribute
            // if attribute is numeric calculate mean
            if (data.attribute(j).isNumeric()) {
                avg[j] = data.attributeStats(j).numericStats.mean;
            // if attribute is nominal find modes
            } else if (data.attribute(j).isNominal()) {
                int[] values = data.attributeStats(j).nominalCounts;
                int max = 0;
                int idxMax = 0;
                for (int k = 0; k < data.attribute(j).numValues(); k++) {
                    if (max < values[k]) {
                        max = values[k];
                        idxMax = k;
                    }
                }
                avg[j]= idxMax;
            }
        }
        return avg;
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
        
        result.append(" ----- FULL DATA ("+ fullNumInstances +")-----\n");
        for (int j = 0; j < centroids.numAttributes(); j++) {
            result.append(centroids.attribute(j).name() + " = ");
            if (centroids.attribute(j).isNominal()) {
                result.append(fullAttrAvg.stringValue(j) + "\n");
            } else {
                result.append(fullAttrAvg.value(j) + "\n");
            }
        }
        result.append("\n");
        for (int i = 0; i < numClusters; i++) {
            result.append(" ----- Cluster #"+ i + " ("+ clusters[i].numInstances()+")-----\n");
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
