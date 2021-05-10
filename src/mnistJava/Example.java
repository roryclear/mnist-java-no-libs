package mnistJava;

import java.io.IOException;
import java.util.HashMap;

public class Example {

	public static void main(String[] args) throws IOException
	{		
		System.out.println("eyup");
		Net n = new Net();
		int[] shape = {784,10,10,10};
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
		
		n.resetNodes();
		}//epoch
		
		
		
	}
	
}
