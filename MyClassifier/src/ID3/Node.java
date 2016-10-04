/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ID3;

import java.util.LinkedHashMap;

/**
 *
 * @author Pipin
 */
public class Node {
	
    private LinkedHashMap<Double, Node> branches;
    private Attribute attribute;
    private int nodeID;

    public Node () {
        this(-1);
    }
    
    public Node (int nodeCounter) {
        branches = new LinkedHashMap<Double, Node>();
        attribute = new Attribute();
        nodeID = nodeCounter;
    }

    public int getNodeID() {
        return nodeID;
    }

    public void setAttribute(Attribute bestAttr) {
        attribute = bestAttr;
    }

    public void addBranch(double value, Node node) {
        branches.put(value, node);
    }

    public LinkedHashMap<Double, Node> getBranches() {
        return branches;
    }

    public Attribute getAttribute() {
        return attribute;
    }
}