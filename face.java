
//  
//
//  Created by Melody Chang on 3/8/17.
//
//


import java.io.*;
import java.util.*;


public class faces {
    
    
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    public double gender;
    public int[] grayness;
    
    
    public Image (int gender){
        this.gender = gender;
    }
    
    //read in training data
    public Image readImage(File file) {
        this.name = file.getName();
        
    }
    
    //read image
    try {
        
        BufferedReader br = new BufferedReader (new FileReader (file));
        Scanner sc = new Scanner(br);
        
        for (int i = 0; i<WIDTH*HEIGHT; i++) { //loop through training set
            grayness[i] = sc.nextInt(); //read integer value for grayness intensity
        }
        
        sc.close;
        br.close;
    
    }
    
    //sigmoid func. assigns double to value between 0 and 1
    private sigmoid (double  num){
        return (double) 1/(1+ Math.exp(-num));
    }
    
    
    
    
    
}



//read input files
//get input, store them in matrix
//set output data female = true (1), male = false (0)
//make two synapses matrices bc 3 layers
//randomly assigned weights to each synapse
//first layer = input data
//matrix multiply each layer and its synapse
//run sigmoid function on matrix to get next layer (which is prediction of output data
//repeat matrix multiplication
//get next layer (which is more refined prediction)
//compare to expected output data using subtraction
//multiply error rate by result of sigmoid function
//backpropagation <-some complicated shit
//get the deltas, update weights
//multiply each layer by a delta







