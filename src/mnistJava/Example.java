package mnistJava;

import java.io.IOException;

public class Example {

	public static void main(String[] args) throws IOException
	{		
		System.out.println("eyup");
		Net n = new Net();
		int[] shape = {784,16,10};
		n.setShape(shape);
		n.setGradsSize(0);
		n.setMomentum(0.5);
		
	//	n.loadWeights();
		
		n.initWeights();
		n.resetNodes();
		
		MnistMatrix[] trainData = n.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		MnistMatrix[] testData = n.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		int[][] odTrainData = n.makeData1D(trainData);
		int[][] odTestData = n.makeData1D(testData);
		
		
		double correct = 0;
		double totalLoss = 0;
		

			
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
			if(i % 1000 == 0)
			{
				double accuracy = (double) correct / i;
				double avgLoss = (double) totalLoss / i;
				
				System.out.println("acc = " + accuracy + "    avgLoss = " + avgLoss);
			}
		}
		
		
		double accuracy = (double) correct / odTrainData.length;
		double avgLoss = (double) totalLoss / odTrainData.length;
		
		System.out.println("acc = " + accuracy + "    avgLoss = " + avgLoss);
		
		
		//testing data
		correct = 0; 
		totalLoss = 0;
		
		for(int i = 0; i < odTestData.length; i++)
		{
			n.resetNodes();
			n.forward(odTestData[i], false);
			if(n.getDigit() == testData[i].getLabel())
			{
				correct +=1;
			}
			totalLoss += n.getLoss(testData[i].getLabel());
		}
		
		avgLoss = (double) totalLoss / testData.length;
		accuracy = (double) correct / testData.length;
		
		System.out.println("TEST acc = " + accuracy + " avgLoss = " + avgLoss);
		
		n.initWeights();
		
		
	}
	
}
