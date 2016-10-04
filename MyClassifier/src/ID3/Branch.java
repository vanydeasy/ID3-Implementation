/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ID3;

/**
 *
 * @author Pipin
 */
public class Branch<V> {
    private Node node;
    private V val;

    public Branch (V val, Node node) {
        this.val = val;
        this.node = node;
    }

    public V getValue() {
        return val;
    }

    public Node getNode() {
        return node;
    }
}