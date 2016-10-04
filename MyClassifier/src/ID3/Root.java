/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ID3;

import java.util.LinkedHashSet;

/**
 *
 * @author Pipin
 */
public class Root extends Node {
    private LinkedHashSet<Branch<Integer>> branches;
    private Attribute attribute;

    public Root() {
        branches = new LinkedHashSet<Branch<Integer>>();
    }
    
    public void setAttribute(Attribute bestAttr) {
        attribute = bestAttr;
    }

    public void addBranch(int value, Node node) {
        Branch branch = new Branch(value, node);
        branches.add(branch);
    }
}