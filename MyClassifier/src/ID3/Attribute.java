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
public class Attribute {
	
    private static String DEFAULT_NAME = "leaf";
    private String name;
    private LinkedHashSet<Double> values;
    private int col;

    public Attribute() {
        name = DEFAULT_NAME;
        values = new LinkedHashSet<Double>();
        col = -1;
    }
    
    public Attribute(String name, LinkedHashSet<Double> values, int col) {
        this.name = name;
        this.values = new LinkedHashSet<Double>(values);
        this.col = col;
    }
    
    public String getName() {
        return name;
    }
    
    public int getCol() {
        return col;
    }

    public LinkedHashSet<Double> getValues() {
        return values;
    }
    
    public void setName(String name) {
        this.name = name;
    }

    public void setValues(LinkedHashSet<Double> values) {
        this.values = new LinkedHashSet<Double> (values);
    }

    public void setCol(int col) {
        this.col = col;
    }

    public int getNumberOfValues() {
        return values.size();
    }
}