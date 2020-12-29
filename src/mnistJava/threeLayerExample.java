package mnistJava;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import javax.imageio.ImageIO;

public class threeLayerExample {
	static MnistMatrix[] trainData;  //training trainData
	static MnistMatrix[] testData; //test trainData
	///LAYER SIZES
 	static int inputSize = 28*28;
	static int hiddenSize = 8;
	static int outputSize = 10;
	static double learningRate = 0.1;
	static int epochs = 100;
	
	//ALL BUT 1 IS BROKEN ATM
	static int batchSize = 1; 
	
	static int randomSamplesDisplayed = 1;

		
	//init nodes
	static double[] layer0nodes = new double[inputSize];
	
	static double[] layer1nodes = new double[hiddenSize];
	
	static double[] layer2nodes = new double[hiddenSize];
	static double[] layer3nodes = new double[outputSize];
	
	//totals for batch
	static double[] layer0nodesTotal = new double[inputSize];
	
	static double[] layer1nodesTotal = new double[hiddenSize];
	
	static double[] layer2nodesTotal = new double[hiddenSize];
	static double[] layer3nodesTotal = new double[outputSize]; 
	
	//weights
	static double[][] hidden0weights = new double[inputSize][hiddenSize];
			
	static double[][] hidden1weights = new double[hiddenSize][hiddenSize];
	static double[][] outputWeights = new double[hiddenSize][outputSize];
	
	//one dimensional image data
	static int[][] odTrainData;
	static int[][] odTestData;
	
	
	public static void main(String[] args) throws IOException
	{
		//ASSIGN RANDOM WEIGHTS OF RANGE -0.1 to 0.1
		
		for(int y = 0; y < hidden0weights.length; y++)
		{
			for(int x = 0; x < hidden0weights[y].length; x++)
			{
				Random r = new Random();
				double w = -0.1 + 0.2 * r.nextDouble();
				hidden0weights[y][x] = w;
			}
		}
		
		
		for(int y = 0; y < hidden1weights.length; y++)
		{
			for(int x = 0; x < hidden1weights[y].length; x++)
			{
				Random r = new Random();
				double w = -0.1 + 0.2 * r.nextDouble();
				hidden1weights[y][x] = w;
			}
		}
		
		
		for(int y = 0; y < outputWeights.length; y++)
		{
			for(int x = 0; x < outputWeights[y].length; x++)
			{
				Random r = new Random();
				double w = -0.1 + 0.2 * r.nextDouble();
				outputWeights[y][x] = w;
			}
		}
		
		//READ data FROM FILES
		trainData = readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		//TEST RANDOM NN
		testNN(trainData, testData);
		
		trainNN(trainData, testData);
		
		testNNwithTestData(trainData, testData);
		
	
		
	}
	
	public static void getOut(int[] data, boolean train)
	{		
		
		for(int x = 0; x < inputSize; x++)
		{
			layer0nodes[x] = sigmoid(data[x]);
			if(train)
			{
				layer0nodesTotal[x] += data[x];
			}
		}
		
		//get layer1 outputs
		for(int x = 0; x < hiddenSize; x++)
		{
			double total = 0;
			for(int y = 0; y < inputSize; y++)
			{
				total += layer0nodes[y]*hidden0weights[y][x];
			}
			total = sigmoid(total);
			layer1nodes[x] = total;
			if(train)
			{
				layer1nodesTotal[x] += total;
			}
		}
		
		//get layer2 outputs
		for(int x = 0; x < hiddenSize; x++)
		{
			double total = 0;
			for(int y = 0; y < hiddenSize; y++)
			{
				total += layer1nodes[y]*hidden1weights[y][x];
			}
			total = sigmoid(total);
			layer2nodes[x] = total;
			if(train)
			{
			layer2nodesTotal[x] += total;
			}
		}
		
		//get layer3 outputs
		for(int x = 0 ; x < outputSize; x++)
		{
			double total = 0;
			for(int y = 0; y < hiddenSize; y++)
			{
				total += layer2nodes[y]*outputWeights[y][x];
			}
			total = sigmoid(total);
			layer3nodes[x] = total;
			if(train)
			{
			layer3nodesTotal[x] += total;
			}
		}
		
	}
	
	public static void testNNWithHanddrawn() throws IOException
	{
		///guess hand drawn by me
		int[] drawn = bmToArray("mnistdata/drawn.bmp");
		
		getOut(drawn,false);
		
		
		
		//get guess and add to histogram
		int guess = getDigit(layer3nodes);
		
		for(int yd = 0; yd < 28; yd++)
		{
			String line = "";
			for(int xd = 0; xd < 28; xd++)
			{
				if(drawn[yd*28 + xd] > 0)
				{
				line+="0";
				}else {
					line+=" ";
				}
			}
			System.out.println(line);
		}
		
		System.out.println("\n----YOU DREW A " + guess + "?------- ");
		String outs = "0 : "+ layer3nodes[0];
		for(int i = 1; i < 10; i++)
		{
			outs+=" , " + i + " : " + layer3nodes[i];
			if(i == 4)
			{
				outs+="\n";
			}
		}
		System.out.println(outs);
	}
	
	public static void testNNwithTestData(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		//create one dimensional images with values 0 - 256??...or 253?
		int[][] odTestData = makeData1D(testData);
		
		System.out.println("\n ----TEST ON TEST trainData------\n");
		HashSet<Integer> randomSamples = new HashSet<>();
		for(int q = 0; q < randomSamplesDisplayed; q++)
		{
			Random r = new Random();
			int random = r.nextInt(10000);
			randomSamples.add(random);
		}
		int correct = 0;
		double loss = 0;
		HashMap<Integer,Integer> hist = new HashMap<Integer, Integer>();
		HashMap<Integer,Integer> correctHist = new HashMap<Integer, Integer>();
		for(int i = 0; i < 10; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}
		
		for(int i = 0; i < testData.length; i++)
		{
			//init nodes
			layer0nodes = new double[inputSize];
			
			layer1nodes = new double[hiddenSize];
			
			layer2nodes = new double[hiddenSize];
			layer3nodes = new double[outputSize]; 
		
			
			getOut(odTestData[i],false);
			
			
			//get guess and add to histogram
			int guess = getDigit(layer3nodes);
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == testData[i].getLabel())
			{
				correctHist.replace(guess, correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(layer3nodes,testData[i].getLabel());
			
			
			if(randomSamples.contains(i))
			{
				displayDigit(testData[i]);
				System.out.println("guess = " + guess + "\n\n");
			}
			
		}
		
		double accuracy = (double) correct/testData.length;
		double avgLoss = (double) loss/testData.length;
		System.out.println("accuracy (test) = " + accuracy);
		System.out.println("avg loss (test) = " + avgLoss);
		System.out.println(hist);	
		System.out.println(correctHist);
		
		///guess hand drawn by me
		int[] output = new int[10];
		for(int i = 0; i < 10; i++)
		{
		//	int[] d = bmToArray("mnistdata/" + i + ".bmp"); //comic sans
			int[] d = bmToArray("mnistdata/" + i + "drawn.bmp");
			for(int x = 0; x < inputSize; x++)
			{
				layer0nodes[x] = sigmoid(d[x]);
			}
			getOut(d,false);
			int guess = getDigit(layer3nodes);
			output[i] = guess;
		}
		
		System.out.println("output for all digits:");
		boolean pass = true; 
		for(int i = 0; i < 10; i++)
		{
			System.out.println(i+": " + output[i]);
			if(output[i] != i)
			{
				pass = false;
			}
		}
		if(pass)
		{
			System.exit(0);
		}
		

		
		
		
	}
	
	
	public static void trainNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		double[][] hw0add = new double[hidden0weights.length][hidden0weights[0].length];
		
		double[][] hw1add = new double[hidden1weights.length][hidden1weights[0].length];
		double[][] owAdd = new double[outputWeights.length][outputWeights[0].length];
		
		//create one dimensional images with values 0 - 256??...or 253?
		int[][] odTrainData = makeData1D(trainData);
		
		System.out.println("\n\n\n TRAINING????? \n\n");
		
		//gonna try training using a batch size of one and 1 epoch???
		//adjust values as it goes???
		for(int z = 0; z < epochs; z++)
		{
		System.out.println("--EPOCH "+z+"-- \n");
		int correct = 0;
		double loss = 0;
		HashMap<Integer,Integer> hist = new HashMap<Integer, Integer>();
		HashMap<Integer,Integer> correctHist = new HashMap<Integer, Integer>();
		for(int i = 0; i < 10; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}

		
		for(int i = 0; i < trainData.length; i++)
		{
			if(i % batchSize == 0)
			{
			layer0nodesTotal = new double[inputSize];
			
			layer1nodesTotal = new double[hiddenSize];
			
			layer2nodesTotal = new double[hiddenSize];
			layer3nodesTotal = new double[outputSize]; 
			}
			//init nodes
			layer0nodes = new double[inputSize];
			
			layer1nodes =new double[hiddenSize];
			
			layer2nodes = new double[hiddenSize];
			layer3nodes = new double[outputSize]; 
			
			if(i % 10000 == 0)
			{
				System.out.println("trainData : " + i +" of " + trainData.length);
				testNNwithTestData(trainData, testData);
				testNNWithHanddrawn();
			}

			
			getOut(odTrainData[i],true);
			
			
			//get guess and add to histogram
			int guess = getDigit(layer3nodes);
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == trainData[i].getLabel())
			{
				correctHist.replace(guess,correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(layer3nodes,trainData[i].getLabel());
			
			double[] expectedOutput = new double[10];
			for(int x = 0; x < 10; x++)
			{
				expectedOutput[x] = 0.0;
				if(x==trainData[i].getLabel())
				{
					expectedOutput[x] = 1.0;
				}
			}
			
			if(i % batchSize == 0)
			{
			
			hw0add = new double[hidden0weights.length][hidden0weights[0].length];	
				
			hw1add = new double[hidden1weights.length][hidden1weights[0].length];
			owAdd = new double[outputWeights.length][outputWeights[0].length];
			
			
			for(int y = 0; y < outputWeights.length; y++)
			{
				double L1Output = layer2nodesTotal[y];
				for(int x = 0; x < outputWeights[y].length; x++)
				{
					///weight between hidden node Y and output node X!!
					double output = layer3nodesTotal[x];
					double expected = expectedOutput[x];
					double dedw = (output - expected)*(output*(1 - output)*(L1Output));
					owAdd[y][x] += dedw;
				}
			}
			
			
			
			//adjust hidden layer 1 weights???
			for(int y = 0; y < hidden1weights.length; y++)
			{
				for(int x = 0; x < hidden1weights[y].length; x++)
				{
					double totalError = 0;
					for(int n = 0; n < layer3nodesTotal.length; n++)	
					{
						totalError += (outputWeights[x][n]*owAdd[x][n])/layer3nodes.length;
					}
					totalError = totalError*(layer2nodesTotal[x]*(1 - layer2nodesTotal[x])*layer1nodesTotal[y]);
					
					
					hw1add[y][x] += totalError;
				}
			}
			
			//adjust hidden layer 0 weights??
			for(int y = 0; y < hidden0weights.length; y++)
			{
				for(int x = 0; x < hidden0weights[y].length; x++)
				{
					double totalError = 0;
					for(int n = 0; n < layer2nodesTotal.length; n++)	
					{
						totalError += ((hidden1weights[x][n]*hw1add[x][n])/layer3nodes.length)/layer2nodes.length;
					}
					totalError = totalError*(layer1nodesTotal[x]*(1 - layer1nodesTotal[x])*layer0nodesTotal[y]);
					
					
					hw0add[y][x] += totalError;
				}
			}
			
			
			for(int y = 0; y < hidden0weights.length; y++)
			{
				for(int x = 0; x < hidden0weights[y].length; x++)
				{
					hidden0weights[y][x] -= (learningRate*hw0add[y][x])/batchSize;
				}
			}
			
			for(int y = 0; y < hidden1weights.length; y++)
			{
				for(int x = 0; x < hidden1weights[y].length; x++)
				{
					hidden1weights[y][x] -= (learningRate*hw1add[y][x])/batchSize;
				}
			}
	
			
			for(int y = 0; y < outputWeights.length; y++)
			{
				for(int x = 0; x < outputWeights[y].length; x++)
				{
					outputWeights[y][x] -= (learningRate*owAdd[y][x])/batchSize;
				}
			}
			
			
			
			
			}
				
			
			//remove
			double accuracy = (double) correct/i;
			if(i % 1000 == 0)
			{
			System.out.println("accuracy (training) = " + accuracy);
			}
			
		}
		
		double accuracy = (double) correct/trainData.length;
		double avgLoss = (double) loss/trainData.length;
		System.out.println("accuracy = " + accuracy);
		System.out.println("avg loss = " + avgLoss);
		System.out.println(hist);
		System.out.println(correctHist);
		
		///guess hand drawn by me
		int[] drawn = bmToArray("mnistdata/drawn.bmp");
		
		getOut(drawn,false);
		
		
		//get guess and add to histogram
		int guess = getDigit(layer3nodes);
		
		
		System.out.println("\n----YOU DREW A " + guess + "?------- ");
		String outs = "0 : " + layer3nodes[0];
		for(int i = 1; i < 10; i++)
		{
			outs+=" , " + i + " : " + layer3nodes[i];
		}
		System.out.println(outs);
		
		
	}//epochs
		
	}

	public static void testNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		//create one dimensional images with values 0 - 256??...or 253?
		int[][] odTrainData = makeData1D(trainData);
		
		
		//TEST NN WITH RANDOM WEIGHTS
		int correct = 0;
		double loss = 0;
		HashMap<Integer,Integer> hist = new HashMap<Integer, Integer>();
		HashMap<Integer,Integer> correctHist = new HashMap<Integer, Integer>();
		for(int i = 0; i < 10; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}
		
		for(int i = 0; i < trainData.length; i++)
		{
			//init nodes
			layer0nodes = new double[inputSize];
			
			layer1nodes = new double[inputSize];
			
			layer2nodes = new double[hiddenSize];
			layer3nodes = new double[outputSize]; 
			
			getOut(odTrainData[i],false);
			
			
			//get guess and add to histogram
			int guess = getDigit(layer3nodes);
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == trainData[i].getLabel())
			{
				correct+=1;
				correctHist.replace(guess, correctHist.get(guess)+1);
			}
			
			//getLoss
			loss+=getLoss(layer3nodes,trainData[i].getLabel());
			
		}
		
		double accuracy = (double) correct/trainData.length;
		double avgLoss = (double) loss/trainData.length;
		System.out.println("accuracy = " + accuracy);
		System.out.println("avg loss = " + avgLoss);
		System.out.println(hist);
		System.out.println(correctHist);
		
		
		System.out.println("\n -----TEST ON MY IMAGE-----\n");
		
		int[] drawn = bmToArray("mnistdata/drawn.bmp");
		
		for(int y = 0; y < 28; y++)
		{
			String line = "";
			for(int x = 0; x < 28; x++)
			{
				int pixel = drawn[y*28 + x]; 
				if(pixel > 0)
				{
					line+="0";
				}else {
					line+=" ";
				}
			}
			System.out.println(line);
		}
		
		
		//init nodes
		layer0nodes = new double[inputSize];
		
		layer1nodes = new double[hiddenSize];
		
		layer2nodes = new double[hiddenSize];
		layer3nodes = new double[outputSize]; 

		getOut(drawn,false);
		
		
		//get guess and add to histogram
		int guess = getDigit(layer3nodes);
		
		System.out.println("\n----RANDOM GUESS is " + guess + "-------");
		
		
	
	}
	
	
	public static int[] bmToArray(String BMPFileName) throws IOException
	{
	int[] out = new int[28*28];
	int index = 0;	
    BufferedImage image = ImageIO.read(new File(BMPFileName));
    for(int y = 0; y < image.getHeight(); y++)
    {
    	for(int x = 0; x < image.getWidth(); x++)
    	{
    		out[index] = -image.getRGB(x, y)/(256*256);	
    		if(out[index] > 255)
    		{
    			out[index] = 255;
    		}	
    		index++;
    	}
    }
	
	return out;
	}
	
	
	
	
	
	
	//calculate loss using output layer and correct answer
	public static double getLoss(double[] output, int answer)
	{
		double loss = 0;
		for(int i = 0; i < output.length; i++)
		{
			if(i == answer)
			{
				loss += (1 - output[i])*(1 - output[i]);
			}else {
				loss += (0 - output[i])*(0 - output[i]);
			}
		}
		return loss;
	}
	
	//returns highest output of output layer (NN's "guess")
	public static int getDigit(double[] outputs)
	{
		int output = 0;
		double largest = outputs[0];
		for(int i = 1; i < outputs.length; i++)
		{
			if(outputs[i] > largest)
			{
				output = i;
				largest = outputs[i];
			}
		}
		return output;
	}
	
	public static double sigmoid(double input)
	{
		   double output = 1 / (1 + Math.exp(-input));
		return output;
	}
	
	
	
	
	public static double relu(double input)
	{
		if(input < 0)
		{
			return 0;
		}
		return input;
	}
	
	
	public static void displayDigit(MnistMatrix trainData)
	{
		for(int r = 0; r < trainData.getNumberOfRows(); r++)
		{
			String row = "";
			for(int c = 0; c < trainData.getNumberOfColumns(); c++)
			{
				if(trainData.getValue(r, c) > 0)
				{
					row = row + "0";
				}else {
					row = row + " ";
				}
			}
			System.out.println(row);
		}
		System.out.println("label = " + trainData.getLabel());
	}
	
	public static void displayData(MnistMatrix[] trainData)
	{
			for(int i = 0; i < trainData.length; i++)
		{
			System.out.println("trainData " + i + " label = " + trainData[i].getLabel());
			for(int r = 0; r < trainData[i].getNumberOfRows(); r++)
			{
				String row = "";
				for(int c = 0; c < trainData[i].getNumberOfColumns(); c++)
				{
					if(trainData[i].getValue(r, c) > 0)
					{
						row = row + " ";
					}else {
						row = row + "0";
					}
				}
				System.out.println(row);
			}
		} 
	}
	
	//converts MnistMatrix[] to int[][]
	//Array of 1D Image trainData instead of 2D
	public static int[][] makeData1D(MnistMatrix[] trainData)
	{
		int[][] out = new int[60000][784];
		for(int i = 0; i < trainData.length; i++)
		{
			int n = 0;
			for(int r = 0; r < trainData[i].getNumberOfRows(); r++)
			{
				for(int c = 0; c < trainData[i].getNumberOfColumns(); c++)
				{
					 out[i][n] = trainData[i].getValue(r, c);
					 n++;
				}
			}
		}
				
		return out;
	}
	
	
	//copied this stuff from stackoverflow
    public static MnistMatrix[] readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        MnistMatrix[] data = new MnistMatrix[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {
            MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols);
            mnistMatrix.setLabel(labelInputStream.readUnsignedByte());
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    mnistMatrix.setValue(r, c, dataInputStream.readUnsignedByte());
                }
            }
            data[i] = mnistMatrix;
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
	
}


	