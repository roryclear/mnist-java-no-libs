package mnistJava;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NetTest {
	MnistMatrix[] testData;
	MnistMatrix[] trainData;
	double[][] odTrainData;
	double[][] odTestData;
	Net n;
	
	@BeforeEach
	void setUp() throws Exception {
		n = new Net();
		int[] shape = {784,16,10};
		n.layers = shape;
		n.learningRate = 0.1;
		n.gradsSize = 0;
		n.momentum = 0.0;
		
		int batchSize = 10;
		
		n.initWeights();
	//	n.loadWeights();
		
		n.resetNodes();
		
		trainData = n.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = n.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		odTrainData = n.makeData1DDouble(trainData);
		odTestData = n.makeData1DDouble(testData);
		
		for(int i = 0; i < odTrainData.length; i++)
		{
			n.forward(odTrainData[i], true);
			n.backward(n.getLoss(trainData[i].getLabel()));
			n.backProp(n.getLoss(trainData[i].getLabel()));
			if(i % batchSize == 0)
			{
				n.optimize();
			}
			n.resetNodes();
		}
	}

	@Test
	void testNet() {
		int correct = 0;
		for(int i = 0; i < odTestData.length; i++)
		{
			n.forward(odTestData[i], false);
			
			if(n.getDigit() == testData[i].getLabel())
			{
				correct += 1;
			}
			n.resetNodes();
		}
		
		System.out.println("correcet = " + correct);
		assertTrue(correct > 6000);
	}

}
