/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.io.IOException;
import java.util.Scanner;
import weka.core.Instances;

/**
 *
 * @author vanyadeasy
 */
public class MyClusterer {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        Scanner scan = new Scanner(System.in);
        System.out.print("filename (.arff or .csv): ");
        String filename = scan.nextLine();

        // Load data from ARFF or CSV
        Instances data = MyClassifier.loadData(filename);
        
        MyAgnes myAgnes = new MyAgnes();
        
        myAgnes.buildClusterer(data);
    }
}
