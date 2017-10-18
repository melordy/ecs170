import java.io.File;
import javax.swing.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Random;
import java.lang.Math;
import java.util.Collections;

public class FaceRecon
{
	public static class Image
	{
		ArrayList<Double> pixels;
		int gender = 0;
		String fname; // file name

		Image(ArrayList<Double> pixels, int gender, String fname)
		{
			this.pixels = pixels;
			this.gender = gender;
			this.fname = fname;
		}
	}

	public static class NeuralNetwork
	{
		int numHidnNodes = 3; // only one hidden layer with 3 hidden nodes
		int imgSize = 128*120; // img size stays constant
		double[][] hiddenWeights = new double[numHidnNodes][imgSize]; // synapse 1
		double[] hiddenUnits = new double[numHidnNodes]; // calculate sum of input*weight
		double[] outputWeights = new double[numHidnNodes]; // synapse 2
		double output = 0; // M or F?
		double eta = 0.5; // learning rate of algoritm

		NeuralNetwork()
		{
			Random rand = new Random();
			for(int i = 0; i < numHidnNodes; i++)
			{
				// initialize weights to random numbers b/w 0 and 0.2
				for(int j = 0; j < imgSize; j++)
				{
					this.hiddenWeights[i][j] = -.1 + rand.nextDouble()*.2;
				}

				this.hiddenUnits[i] = 0;
				this.outputWeights[i] = -.1 + rand.nextDouble()*.2;
			}
		}

		// train the algorithm
		void train(Image img)
		{
			ArrayList<Double> inputNodes = img.pixels; // input data
			// each val is sum of a node * all edge weights into hidden layer
			// (node has edge to each hidden node in next layer)
			ArrayList<Double> hiddenSums = new ArrayList<Double>(); 
			// input layer to the hidden layer
			for(int i = 0; i < numHidnNodes; i++)
			{
				hiddenUnits[i] = 0;
				for(int j = 0; j < inputNodes.size(); j++)
				{
					// add node * edge weight (matrix multiplication)
					hiddenUnits[i] += (inputNodes.get(j)*hiddenWeights[i][j]);
				}
				// add entire sum for node to hiddenSums
				hiddenSums.add(hiddenUnits[i]);
				// run sigmoid function to get the hidden nodes (the next layer)
				hiddenUnits[i] = sigmoid(hiddenUnits[i]);
			}

			double sum = 0.0;
			// hidden layer to the output layer			
			for(int i = 0; i < numHidnNodes; i++)
			{
				// add node * edge weight (matrix multiplication)
				sum += (hiddenUnits[i]*outputWeights[i]);
			}
			// run sigmoid function to get the output nodes (the next layer)
			output = sigmoid(sum);

			// backpropagation
			if(output != img.gender)
			{
				double outputError = img.gender - output;
				double[] delta = new double[numHidnNodes];
				//update weights = gradient descent
				for(int i = 0; i < numHidnNodes; i++)
				{
					delta[i] = outputError * outputWeights[i];
					for(int j = 0; j < numHidnNodes; j++)
					{
						hiddenWeights[i][j] = eta*delta[i]*inputNodes.get(j)*sigmoidPrime(hiddenSums.get(i));
					}
					outputWeights[i] = eta*outputError*hiddenUnits[i]*sigmoidPrime(sum);
				} 
			}
		}

		// calculate confidence
		double confidence(double value) {
		    return Math.abs(.5 - value) * 2;
		}

		// run through neural network to test algorithm
		int test(Image img)
		{
			// input to the hidden layer
			ArrayList<Double> inputNodes = img.pixels; // input data
			for(int i = 0; i < numHidnNodes; i++)
			{
				hiddenUnits[i] = 0;
				for(int j = 0; j < inputNodes.size(); j++)
				{
					hiddenUnits[i] += (inputNodes.get(j)*hiddenWeights[i][j]);
				}
				hiddenUnits[i] = sigmoid(hiddenUnits[i]);
			}			

			double sum = 0.0;
			// hidden layer to the output layer			
			for(int i = 0; i < numHidnNodes; i++)
			{
				// add node * edge weight (matrix multiplication)
				sum += (hiddenUnits[i]*outputWeights[i]);
			}
			// run sigmoid function to get the output nodes (the next layer)
			output = sigmoid(sum);

			if (output >= 0.50) // 0.50-1.00 is MALE image
			{
                System.out.println(img.fname + " MALE " + confidence(output) + ".");
                return 1;
            } 
            else // 0.00- 0.49 is FEMALE image
            {
                System.out.println(img.fname + " FEMALE " + confidence(output) + ".");
                return 0;
            }
		}
	}

	// sigmoid function
	public static double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    // derivative of sigmoid function
    public static double sigmoidPrime(double x) {
        return Math.pow(Math.E,-1*x)/Math.pow(1+Math.pow(Math.E,-1*x), 2);
    }

    // load test directory
    public static ArrayList<Image> load(final String testDir)
    {
    	System.out.println("loading testDir");
    	final ArrayList<Image> testFiles = new ArrayList<Image>();
    	File[] files = new File(testDir).listFiles();
    	for(File f:files)
    	{
    		try
    		{
    			Scanner sc1 = new Scanner(new File(testDir + f.getName()));
    			// puts all numbers in file into a list as doubles
    			ArrayList<Double> doubleVals = new ArrayList<Double>();
    			while(sc1.hasNextLine())
    			{
    				Scanner sc2 = new Scanner(sc1.nextLine());
    				while(sc2.hasNextLine())
    				{
    					// making all values doubles less than 1 (divide by 255)
    					doubleVals.add(Double.parseDouble(sc2.next())/255.0);
    				}
    			}
    			try
    			{
    				testFiles.add(new Image(doubleVals, -1, f.getName()));
    			}
    			catch(Exception e)
    			{
    				e.printStackTrace();
    			}
    		}
    		catch(Exception e)
    		{
    			e.printStackTrace();
    		}
    	}
    	return testFiles;
    }

    // load female and male directories
    public static ArrayList<Image> load(final String femaleDir, final String maleDir)
    {
    	System.out.println("loading femaleDir and maleDir");
    	final ArrayList<Image> fmFiles = new ArrayList<Image>();
    	File[] femaleFiles = new File(femaleDir).listFiles();
    	File[] maleFiles = new File(maleDir).listFiles();
    	for(File f:femaleFiles)
    	{
    		System.out.println("looping through femaleFiles");
    		try
    		{
    			System.out.println("initializing sc1");
    			Scanner sc1 = new Scanner(new File(femaleDir + f.getName()));
    			System.out.println("initialized sc1");
    			// puts all numbers in file into a list as doubles
    			ArrayList<Double> doubleVals = new ArrayList<Double>();
    			while(sc1.hasNextLine())
    			{
    				System.out.println("Reading from sc1");
    				Scanner sc2 = new Scanner(sc1.nextLine());
    				while(sc2.hasNextLine())
    				{
    					System.out.println("Reading from sc2");
    					// making all values doubles less than 1 (divide by 255)
    					doubleVals.add(Double.parseDouble(sc2.next())/255.0);
    				}
    			}
    			try
    			{
    				fmFiles.add(new Image(doubleVals, 0, f.getName()));
    			}
    			catch(Exception e)
    			{
    				e.printStackTrace();
    			}
    		}
    		catch(Exception e)
    		{
    			e.printStackTrace();
    		}
    	}

    	for(File f:maleFiles)
    	{
    		try
    		{
    			Scanner sc1 = new Scanner(new File(maleDir + f.getName()));
    			// puts all numbers in file into a list as doubles
    			ArrayList<Double> doubleVals = new ArrayList<Double>();
    			while(sc1.hasNextLine())
    			{
    				Scanner sc2 = new Scanner(sc1.nextLine());
    				while(sc2.hasNextLine())
    				{
    					// making all values doubles less than 1 (divide by 255)
    					doubleVals.add(Double.parseDouble(sc2.next())/255.0);
    				}
    			}
    			try
    			{
    				fmFiles.add(new Image(doubleVals, 1, f.getName()));
    			}
    			catch(Exception e)
    			{
    				e.printStackTrace();
    			}
    		}
    		catch(Exception e)
    		{
    			e.printStackTrace();
    		}
    	}

    	Collections.shuffle(fmFiles);
    	return fmFiles;
    }

	public static void main(String[] args)
    {
    	boolean train = false;
    	boolean test = false;
    	String femaleDir = "";
    	String maleDir = "";
    	String testDir = "";
    	int index = 0;

		// PART 1 & 3 OF EVALUATING PERFORMANCE
    	// looping through command line arguments
    	while(index < args.length)
    	{
    		System.out.println(args[index]);
    		// usage: -train female male
    		if(args[index].equalsIgnoreCase("-train"))
    		{
    			train = true;
    			femaleDir = args[index+1];
    			System.out.println(femaleDir);
    			maleDir = args[index+2];
    			System.out.println(maleDir);
    			index+=3;
    			System.out.println("training");
    		}
    		else if(args[index].equalsIgnoreCase("-test"))
    		{
    			test = true;
    			testDir = args[index+1];
    			index+=2;
    			System.out.println("testing");
    		}
    		else
    		{
    			index+=1;
    		}
    	}

    	if(train && test)
    	{
    		System.out.println("Train and test");
    		ArrayList<Image> fmFiles = load(femaleDir, maleDir);
    		NeuralNetwork nn = new NeuralNetwork();

    		// Training
    		int iter = 0;
    		while(iter < 32)
    		{
	    		for(Image img:fmFiles)
	    		{
	    			nn.train(img);
	    		}
	    		iter++;
	    	}

	    	// Testing
	    	ArrayList<Image> testFiles = load(testDir);
	    	for(Image img:testFiles)
	    	{
	    		nn.test(img);
	    	}
	    }
	    // PART 2 OF EVALUATING PERFORMANCE
	    else if(train)
	    {
	    	System.out.println("PART2");
    		// split into 5 folds
    		ArrayList<Image> fmFiles = load(femaleDir, maleDir);
    		System.out.println("fmFiles loaded");

    		ArrayList<ArrayList<Image>> folds = new ArrayList<ArrayList<Image>>();
    		int splitSize = fmFiles.size()/5;
    		for(int i = 0; i < 5; i++)
    		{
    			ArrayList<Image> temp 
    				= new ArrayList<Image>(fmFiles.subList(i*splitSize, (i+1)*splitSize));
    			folds.add(temp);
    		}

    		// perform 5-fold cross-validation
    		NeuralNetwork nn = new NeuralNetwork();
    		double avg = 0;
			for(int i = 0; i < 5; i++)
    		{
    			// training neural network
    			int iter = 0;
    			while(iter < 32)
    			{
    				for(int j = 0; j < 5; j++)
    				{
    					for(int k = 0; k < splitSize; k++)
    					{
    						nn.train(folds.get(j).get(k));
    					}
    				}
    				iter++;
    			}

    			// testing neural network
    			int nCorrect = 0;
    			for(int j = 0; j < splitSize; j++)
    			{
    				if(nn.test(folds.get(i).get(j)) == folds.get(i).get(j).gender)
    				{
    					nCorrect++;
    				}
    			}

    			System.out.println("Percent Correct: " + nCorrect/(folds.get(i).size()+0.0));
                avg += nCorrect/(folds.get(i).size()+0.0); 
            }
            System.out.println("AVERAGE : " + avg/5.0);
	    }
	    else
	    {
	    	System.out.println("Please train before testing");
	    }

	    return;
    }
}