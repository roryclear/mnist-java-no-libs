package mnistJava;

import java.io.IOException;

public class Example {

	public static void main(String[] args) throws IOException
	{		
		System.out.println("eyup");
		Net n = new Net();
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
			if(n.getDigit() == trainData[i].getLabel())
			{
				correct += 1;
				System.out.println("correct " + n.getDigit());
			}
		}
		
		
		n.initWeights();
		
		
	}
	
}
