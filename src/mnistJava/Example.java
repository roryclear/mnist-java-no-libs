package mnistJava;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import javax.imageio.ImageIO;

public class Example {
	static MnistMatrix[] trainData;
	static MnistMatrix[] testData;
	
	static double[][] odTrainData;
	static double[][] odTrainDataBackup;
	static double[][] odTestData;
	
	static double avgLoss;
	static double accuracy;

	static int batchSize = 10;
	
	static boolean shuffle = false;
	
	static boolean augment = false;
	
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
		
		odTrainDataBackup = odTrainData;
		
		int showTrainAccuracyInterval = 10000;
		int epochs = 10;
		
		double correct = 0;
		double totalLoss = 0;
		
		testNN(n);
		double lowestLoss = avgLoss;
		
		
		//shuffled index 
		ArrayList<Integer> shuffled = new ArrayList<>();
		if(shuffle)
		{
			for(int i = 0; i < trainData.length; i++)
			{
				shuffled.add(i);
			}
		}
		
		for(int epoch = 0; epoch < epochs; epoch++)
		{
		if(shuffle)
		{
			Collections.shuffle(shuffled);
		}
		
		if(augment)
		{
			odTrainData = odTrainDataBackup;
			for(int i = 0; i < odTrainData.length; i++)
			{
				odTrainData[i] = augmentArray(odTrainData[i]);
			}
		}
		
		correct = 0;
		totalLoss = 0;
		System.out.println("\n--------EPOCH " + epoch + "--------");
		
		n.resetNodes();
		for(int i = 0; i < odTrainData.length; i++)
		{
			int index = i;
			if(shuffle)
			{
				index = shuffled.get(i);
			}
			
			n.forward(odTrainData[index], true);
			n.backward(n.getLoss(trainData[index].getLabel()));
			if(i % batchSize == 0)
			{
			n.optimize();
			}
			//n.backProp(n.getLoss(trainData[i].getLabel()));
			if(n.getDigit() == trainData[index].getLabel())
			{
				correct += 1;
			}
			totalLoss += n.getTotalOutputLoss(trainData[index].getLabel());
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
	//		n.saveWeights();
		}
		
		n.resetNodes();
		}//epoch
		
		
		
	}
	
	public static double[] augmentArray(double[] array)
	{
		//make2d
		double[][] array2d = new double[(int) Math.sqrt(array.length)][(int) Math.sqrt(array.length)];
		for(int y = 0; y < array2d.length; y++)
		{
			for(int x = 0; x < array2d[0].length; x++)
			{
				array2d[y][x] = array[x + y*28];
			}
		}
		
		array2d = rotateArray(array2d,90);
		
		//make1d
		for(int y = 0; y < array2d.length; y++) {
			for(int x = 0; x < array2d[0].length; x++)
			{
				array[x + y*28] = array2d[x][y];
			}
		}
		
		return array;
	}
	
	public static double[][] rotateArray(double[][] array, int degrees)
	{
		return array;
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
	    
	    n.resetNodes();
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
