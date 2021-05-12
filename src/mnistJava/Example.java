package mnistJava;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

import javax.imageio.ImageIO;

public class Example {

	public static void main(String[] args) throws IOException
	{		
		System.out.println("eyup");
		Net n = new Net();
		int[] shape = {784,32,10,10};
		n.setShape(shape);
		n.setLearningRate(0.1);
		n.setGradsSize(0);
		n.setMomentum(0.5);
		
	//	n.loadWeights();
		
		n.initWeights();
		n.resetNodes();
		
		MnistMatrix[] trainData = n.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		MnistMatrix[] testData = n.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		int[][] odTrainData = n.makeData1D(trainData);
		int[][] odTestData = n.makeData1D(testData);
		
		int epochs = 100;
		
		double correct = 0;
		double totalLoss = 0;
		
		
		for(int z = 0; z < epochs; z++)
		{
		correct = 0;
		totalLoss = 0;
		System.out.println("--------EPOCH " + z + "--------");
		
			
		for(int i = 0; i < odTrainData.length; i++)
		{
			n.forward(odTrainData[i], true);
			n.backProp(trainData[i].getLabel());
			if(n.getDigit() == trainData[i].getLabel())
			{
				correct += 1;
			}
			totalLoss += n.getLoss(trainData[i].getLabel());
			n.resetNodes();
			
			//delete
			if(i % 1000 == 0 && i > 0)
			{
				double accuracy = (double) correct / (i+1);
				double avgLoss = (double) totalLoss / (i+1);
				
				System.out.println("acc = " + accuracy + "    avgLoss = " + avgLoss);
			}
		}
		
		
		double accuracy = (double) correct / odTrainData.length;
		double avgLoss = (double) totalLoss / odTrainData.length;
		
		System.out.println("acc = " + accuracy + "    avgLoss = " + avgLoss);
		
		
		//testing data
		correct = 0; 
		totalLoss = 0;
		
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
			totalLoss += n.getLoss(testData[i].getLabel());
		}
		
		avgLoss = (double) totalLoss / testData.length;
		accuracy = (double) correct / testData.length;
		
		System.out.println("TEST acc = " + accuracy + " avgLoss = " + avgLoss);
		System.out.println(guesses);
		System.out.println(correctGuesses);
		
		//0 - 9 test
		int[] output = new int[10];
		
		for(int i = 0; i < 10; i++)
		{
		int xOffset = (i % 5) * 29;
		int yOffset = (i / 5) * 29;
			
		int[] digitArray = new int[28*28];
		int index = 0;	
	    BufferedImage image = ImageIO.read(new File("mnistdata/digitsGrid.bmp"));
	    for(int y = 0; y < 28; y++)
	    {
	    	for(int x = 0; x < 28; x++)
	    	{
	    		digitArray[index] = -image.getRGB(xOffset + x, yOffset + y)/(256*256);	
	    		if(digitArray[index] > 255)
	    		{
	    			digitArray[index] = 255;
	    		}	
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
			System.out.println(i + " -> " + output[i]);
			if(output[i] == desiredOutput[i])
			{
				correct +=1;
			}
		}
		
		accuracy = (double) correct / output.length;
		
		System.out.println("0-9 accuracy = " + accuracy);
		if(Arrays.equals(desiredOutput, output))
		{
			System.exit(0);	
		}
		
		
		//0-9
		
		
		n.resetNodes();
		}//epoch
		
		
		
	}
	
}
