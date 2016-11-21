/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

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
    private DistanceFunction distanceFunction = null;
    private int[] dataCluster = null;
    private int numClusters = 2;
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        distanceFunction = new EuclideanDistance(instances);
        // Initiate all clusters (one cluster per instance)
        dataCluster = new int[instances.numInstances()];
        for(int i=0;i<instances.numInstances();i++) dataCluster[i] = i;
        
        double[][] distance = countDistance(instances);
        
        
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

    public void setNumClusters(int num) {
        numClusters = num;
    }
    
    public double[][] countDistance(Instances instances) {
        int nInstances = instances.numInstances();
        double[][] distanceMatrix = null;
        distanceMatrix = new double[nInstances][nInstances];
        
        for (int i = 0; i < nInstances; i++) {
            distanceMatrix[i][i] = 0;
            for (int j = i + 1; j < nInstances; j++) {
                distanceMatrix[i][j] = distanceFunction.distance(instances.instance(i), instances.instance(j));
                distanceMatrix[j][i] = distanceMatrix[i][j];
            }
        }
        return distanceMatrix;
    }
    
}
