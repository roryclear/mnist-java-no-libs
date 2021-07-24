package mnistJava;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

import javax.imageio.ImageIO;

public class Example {
	static MnistMatrix[] trainData;
	static MnistMatrix[] testData;
	
	static double[][] odTrainData;
	static double[][] odTestData;
	
	static double avgLoss;
	static double accuracy;

	public static void main(String[] args) throws IOException
	{	
		Net n = new Net();
		int[] shape = {784,16,10};
		n.layers = shape;
		n.learningRate = 0.1;
		n.gradsSize = 0;
		n.momentum = 0.0;
			
		n.initWeights();
	//	n.loadWeights();
		
		n.resetNodes();
		
		trainData = n.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = n.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		odTrainData = n.makeData1DDouble(trainData);
		odTestData = n.makeData1DDouble(testData);
		
		int showTrainAccuracyInterval = 10000;
		int epochs = 10;
		
		double correct = 0;
		double totalLoss = 0;
		
		testNN(n);
		double lowestLoss = avgLoss;
		
		for(int epoch = 0; epoch < epochs; epoch++)
		{
		correct = 0;
		totalLoss = 0;
		System.out.println("\n--------EPOCH " + epoch + "--------");
		
			
		for(int i = 0; i < odTrainData.length; i++)
		{
			n.forward(odTrainData[i], true);
			n.backProp(n.getLoss(trainData[i].getLabel()));
			
			if(n.getDigit() == trainData[i].getLabel())
			{
				correct += 1;
			}
			totalLoss += n.getTotalOutputLoss(trainData[i].getLabel());
			n.resetNodes();
			
			//delete
			if(i % showTrainAccuracyInterval == 0 && i > 0)
			{
				accuracy = (double) correct / (i+1);
				avgLoss = (double) totalLoss / (i+1);
				
				System.out.println(i + "/" + trainData.length + ": acc = " + accuracy + "    avgLoss = " + avgLoss);
			}
		}
		
		
		accuracy = (double) correct / odTrainData.length;
		avgLoss = (double) totalLoss / odTrainData.length;
		
		System.out.println(trainData.length + "/" + trainData.length +  ": acc = " + accuracy + "    avgLoss = " + avgLoss);
		
		
		
		testNN(n);

		if(avgLoss < lowestLoss)
		{
			lowestLoss = avgLoss;
			n.saveWeights();
		}
		
		n.resetNodes();
		}//epoch
		
		
		
	}
	
	public static void testNN(Net n) throws IOException
	{
		//testing data
		double correct = 0; 
		double totalLoss = 0;
		
		HashMap<Integer, Integer> guesses = new HashMap<>();
		HashMap<Integer, Integer> correctGuesses = new HashMap<>();
		for(int i = 0; i < 10; i++)
		{
			guesses.put(i, 0);
			correctGuesses.put(i, 0);
		}
		
		for(int i = 0; i < odTestData.length; i++)
		{
			n.resetNodes();
			n.forward(odTestData[i], false);
			guesses.put(n.getDigit(),guesses.get(n.getDigit())+1); //inc
			if(n.getDigit() == testData[i].getLabel())
			{
				correct +=1;
				correctGuesses.put(n.getDigit(),correctGuesses.get(n.getDigit())+1); //inc
			}
			totalLoss += n.getTotalOutputLoss(testData[i].getLabel());
		}
		
		avgLoss = (double) totalLoss / testData.length;
		accuracy = (double) correct / testData.length;
		
		System.out.println("\n\nTEST acc = " + accuracy + " avgLoss = " + avgLoss);
		System.out.println(guesses);
		System.out.println(correctGuesses);
	
		
		//0 - 9 test
		int[] output = new int[10];
		
		for(int i = 0; i < 10; i++)
		{
		int xOffset = (i % 5) * 29;
		int yOffset = (i / 5) * 29;
			
		double[] digitArray = new double[28*28];
		int index = 0;	
	    BufferedImage image = ImageIO.read(new File("mnistdata/digitsGrid.bmp"));
	    for(int y = 0; y < 28; y++)
	    {
	    	for(int x = 0; x < 28; x++)
	    	{
	    		digitArray[index] = Double.valueOf(-image.getRGB(xOffset + x, yOffset + y)/(256*256)) / 255;		
	    		index++;
	    	}
	    }
	    
	    n.forward(digitArray, false);
		output[i] = n.getDigit();
	    
		}
		
		System.out.println("\n\nzero to nine output:");
		int[] desiredOutput = {0,1,2,3,4,5,6,7,8,9};
		accuracy = 0;
		correct = 0;
		for(int i = 0; i < 10; i++)
		{
			System.out.println(desiredOutput[i] + " -> " + output[i]);
			if(output[i] == desiredOutput[i])
			{
				correct +=1;
			}
		}
		
		accuracy = (double) correct / output.length;
		
		System.out.println("0-9 accuracy = " + accuracy);
		if(Arrays.equals(desiredOutput, output))
		{
	//		System.exit(0);	
		}
		
		
		//0-9
	}
	
}
