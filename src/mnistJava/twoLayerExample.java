package mnistJava;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;

import javax.imageio.ImageIO;


public class twoLayerExample {
	static MnistMatrix[] trainData;  
	static MnistMatrix[] testData; 
	///LAYER SIZES
 	static int inputSize = 28*28;
	static int hiddenSize = 512; //512
	static int outputSize = 10;
	static double learningRate = 0.1;
	static int epochs = 100; //100
	static double randomWeightRange = 0.1;
	
	static int randomSamplesDisplayed = 1;
	static int testNNevery = 10000; //10000
	static int showTrainingAccEvery = 1000;
	
	
	//init nodes
	static double[] layer0nodes = new double[inputSize];
	static double[] layer1nodes = new double[hiddenSize];
	static double[] layer2nodes = new double[outputSize];
	
	//totals for bp
	static double[] layer0nodesTotal = new double[inputSize];
	static double[] layer1nodesTotal = new double[hiddenSize];
	static double[] layer2nodesTotal = new double[outputSize]; 
	
	//weights
	static double[][] hiddenWeights = new double[inputSize][hiddenSize];
	static double[][] outputWeights = new double[hiddenSize][outputSize];
	
	//one dimensional data
	static int[][] odTrainData;
	static int[][] odTestData;
	
	//save and load weights
	static boolean saveWeights = false;
	static boolean loadWeights = false;
	
	static String saveFile = "2layerWeights.txt";
	static String loadFile = "2layerWeights.txt";
	
	public static void main(String[] args) throws IOException
	{
		if(loadWeights)
		{
			loadWeights();
		}else {
			makeRandomWeights();
		}
		//READ data FROM FILES
		trainData = readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		testNN(trainData, testData);
		
		trainNN(trainData, testData);
		
		testNN(trainData, testData);
		
	
		
	}
	
	public static void forward(int[] data, boolean train)
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
				total += layer0nodes[y]*hiddenWeights[y][x];
			}
			total = sigmoid(total);
			layer1nodes[x] = total;
			if(train)
			{
			layer1nodesTotal[x] += total;
			}
		}
		
		//get layer2 outputs
		for(int x = 0 ; x < outputSize; x++)
		{
			double total = 0;
			for(int y = 0; y < hiddenSize; y++)
			{
				total += layer1nodes[y]*outputWeights[y][x];
			}
			total = sigmoid(total);
			layer2nodes[x] = total;
			if(train)
			{
			layer2nodesTotal[x] += total;
			}
		}
		
	}
	
	public static void testNNWithHanddrawn() throws IOException
	{
		///guess hand drawn by me
		int[] drawn = bmToArray("mnistdata/drawn.bmp");
		
		forward(drawn,false);
		
		
		
		//get guess and add to histogram
		int guess = getDigit(layer2nodes);
		
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
		String outs = "0 : "+ layer2nodes[0];
		for(int i = 1; i < outputSize; i++)
		{
			outs+=" , " + i + " : " + layer2nodes[i];
			if(i == 4)
			{
				outs+="\n";
			}
		}
		System.out.println(outs);
	}
	
	public static void testNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		//create one dimensional images with values 0 - 255
		int[][] odTestData = makeData1D(testData);
		
		System.out.println("\n ----TEST ON TEST DATA------\n");
		HashSet<Integer> randomSamples = new HashSet<>();
		for(int q = 0; q < randomSamplesDisplayed; q++)
		{
			Random r = new Random();
			int random = r.nextInt(testNNevery);
			randomSamples.add(random);
		}
		int correct = 0;
		double loss = 0;
		HashMap<Integer,Integer> hist = new HashMap<Integer, Integer>();
		HashMap<Integer,Integer> correctHist = new HashMap<Integer, Integer>();
		for(int i = 0; i < outputSize; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}
		
		for(int i = 0; i < testData.length; i++)
		{
			resetNodes();
		
			
			forward(odTestData[i],false);
			
			
			//get guess and add to histogram
			int guess = getDigit(layer2nodes);
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == testData[i].getLabel())
			{
				correctHist.replace(guess, correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(layer2nodes,testData[i].getLabel());
			
			
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
		int[] output = new int[outputSize];
		for(int i = 0; i < outputSize; i++)
		{
		//	int[] d = bmToArray("mnistdata/" + i + ".bmp"); //comic sans
			int[] d = bmToArray("mnistdata/" + i + "drawn.bmp");
			forward(d,false);
			int guess = getDigit(layer2nodes);
			output[i] = guess;
		}
		
		System.out.println("output for all digits:");
		boolean pass = true; 
		for(int i = 0; i < outputSize; i++)
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
	
	public static void makeRandomWeights()
	{
		for(int y = 0; y < hiddenWeights.length; y++)
		{
			for(int x = 0; x < hiddenWeights[y].length; x++)
			{
				Random r = new Random();
				double w = -randomWeightRange + 2 * randomWeightRange * r.nextDouble();
				hiddenWeights[y][x] = w;
			}
		}
		
		
		for(int y = 0; y < outputWeights.length; y++)
		{
			for(int x = 0; x < outputWeights[y].length; x++)
			{
				Random r = new Random();
				double w = -randomWeightRange + 2 * randomWeightRange * r.nextDouble();
				outputWeights[y][x] = w;
			}
		}
	}
	
	public static void loadWeights()
	{
	    try {
	        File myObj = new File(loadFile);
	        Scanner myReader = new Scanner(myObj);
	        myReader.nextLine();
	        int y = 0;
	        while (myReader.hasNextLine()) {
	          String data = myReader.nextLine();
	          String[] weightStrings = data.split(",");
	          if(y < hiddenWeights.length)
	          {
	        	  for(int x = 0; x < weightStrings.length; x++)
	        	  {
	        		  hiddenWeights[y][x] = Double.parseDouble(weightStrings[x]);
	        	  }
	          }else{
	        	  for(int x = 0; x < weightStrings.length; x++)
	        	  {
	        		  outputWeights[y - hiddenWeights.length][x] = Double.parseDouble(weightStrings[x]);
	        	  }
	          }
	          y++;
	        }
	        myReader.close();
	      } catch (FileNotFoundException e) {
	        System.out.println("An error occurred.");
	        e.printStackTrace();
	      }
	}
	
	public static void saveWeights()
	{
	    try {
	        FileWriter myWriter = new FileWriter(saveFile);
	        myWriter.append("hiddenSize = " + hiddenSize);
	        for(int y = 0; y < hiddenWeights.length; y++)
	        {
	        	String line = ""+hiddenWeights[y][0];
	        	for(int x = 1; x < hiddenWeights[y].length; x++)
	        	{
	        		line = line + "," + hiddenWeights[y][x];
	        	}
	        	myWriter.append("\n" + line);
	        }
	        for(int y = 0; y < outputWeights.length; y++)
	        {
	        	String line = ""+outputWeights[y][0];
	        	for(int x = 1; x < outputWeights[y].length; x++)
	        	{
	        		line = line + "," + outputWeights[y][x];
	        	}
	        	myWriter.append("\n" + line);
	        }
	        
	        
	        myWriter.close();
	        System.out.println("weights saved");
	      } catch (IOException e) {
	        System.out.println("An error occurred.");
	        e.printStackTrace();
	      }
	}
	
	public static void resetNodes()
	{
		layer0nodes = new double[inputSize];
		layer1nodes = new double[hiddenSize];
		layer2nodes = new double[outputSize];
		
		layer0nodesTotal = new double[inputSize];
		layer1nodesTotal = new double[hiddenSize];
		layer2nodesTotal = new double[outputSize]; 
	}
	
	public static void trainNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		
		double[][] hwGrads = new double[hiddenWeights.length][hiddenWeights[0].length];
		double[][] owGrads = new double[outputWeights.length][outputWeights[0].length];
		
		//create one dimensional images with values 0 - 255
		int[][] odTrainData = makeData1D(trainData);
		
		System.out.println("\n\n\n TRAINING????? \n\n");
	
	
		for(int z = 0; z < epochs; z++)
		{
		if(saveWeights)
		{
		saveWeights();
		}
		System.out.println("--EPOCH "+z+"-- \n");
		int correct = 0;
		double loss = 0;
		HashMap<Integer,Integer> hist = new HashMap<Integer, Integer>();
		HashMap<Integer,Integer> correctHist = new HashMap<Integer, Integer>();
		for(int i = 0; i < outputSize; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}

		
		for(int i = 0; i < trainData.length; i++)
		{
			resetNodes();
			
			if(i % testNNevery == 0)
			{
				System.out.println("data : " + i +" of " + trainData.length);
				testNN(trainData, testData);
				testNNWithHanddrawn();
			}

			
			forward(odTrainData[i],true);
			
			
			//get guess and add to histogram
			int guess = getDigit(layer2nodes);
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == trainData[i].getLabel())
			{
				correctHist.replace(guess,correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(layer2nodes,trainData[i].getLabel());
			
			double[] expectedOutput = new double[outputSize];
			for(int x = 0; x < outputSize; x++)
			{
				expectedOutput[x] = 0.0;
				if(x==trainData[i].getLabel())
				{
					expectedOutput[x] = 1.0;
				}
			}
			
				
			hwGrads = new double[hiddenWeights.length][hiddenWeights[0].length];
			owGrads = new double[outputWeights.length][outputWeights[0].length];	
			
			for(int y = 0; y < outputWeights.length; y++)
			{
				double L1Output = layer1nodesTotal[y];
				for(int x = 0; x < outputWeights[y].length; x++)
				{
					double output = layer2nodesTotal[x];
					double expected = expectedOutput[x];
					double dedw = (output - expected)*(output*(1 - output)*(L1Output));
					owGrads[y][x] += dedw;
				}
			}
			
			
			
			//adjust hidden layer weights???
			for(int y = 0; y < hiddenWeights.length; y++)
			{
				for(int x = 0; x < hiddenWeights[y].length; x++)
				{
					double totalError = 0;
					for(int n = 0; n < layer2nodesTotal.length; n++)	
					{
						totalError += (outputWeights[x][n]*owGrads[x][n])/layer2nodes.length;
					}
					totalError = totalError*(layer1nodesTotal[x]*(1 - layer1nodesTotal[x])*layer0nodesTotal[y]);
					
					
					hwGrads[y][x] += totalError;
				}
			}
			
			
			for(int y = 0; y < hiddenWeights.length; y++)
			{
				for(int x = 0; x < hiddenWeights[y].length; x++)
				{
					hiddenWeights[y][x] -= (learningRate*hwGrads[y][x]);
				}
			}
	
		
			for(int y = 0; y < outputWeights.length; y++)
			{
				for(int x = 0; x < outputWeights[y].length; x++)
				{
					outputWeights[y][x] -= (learningRate*owGrads[y][x]);
				}
			}
							
			
			//remove
			double accuracy = (double) correct/i;
			if(i % showTrainingAccEvery == 0)
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
		
		forward(drawn,false);
		
		
		//get guess and add to histogram
		int guess = getDigit(layer2nodes);
		
		
		System.out.println("\n----YOU DREW A " + guess + "?------- ");
		String outs = "0 : " + layer2nodes[0];
		for(int i = 1; i < outputSize; i++)
		{
			outs+=" , " + i + " : " + layer2nodes[i];
		}
		System.out.println(outs);
		
		
	}//epochs
		
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
	
	
	public static void displayDigit(MnistMatrix data)
	{
		for(int r = 0; r < data.getNumberOfRows(); r++)
		{
			String row = "";
			for(int c = 0; c < data.getNumberOfColumns(); c++)
			{
				if(data.getValue(r, c) > 0)
				{
					row = row + "0";
				}else {
					row = row + " ";
				}
			}
			System.out.println(row);
		}
		System.out.println("label = " + data.getLabel());
	}
	
	public static void displayData(MnistMatrix[] data)
	{
			for(int i = 0; i < data.length; i++)
		{
			System.out.println("data " + i + " label = " + data[i].getLabel());
			for(int r = 0; r < data[i].getNumberOfRows(); r++)
			{
				String row = "";
				for(int c = 0; c < data[i].getNumberOfColumns(); c++)
				{
					if(data[i].getValue(r, c) > 0)
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
	//Array of 1D Image data instead of 2D
	public static int[][] makeData1D(MnistMatrix[] data)
	{
		int[][] out = new int[60000][784];
		for(int i = 0; i < data.length; i++)
		{
			int n = 0;
			for(int r = 0; r < data[i].getNumberOfRows(); r++)
			{
				for(int c = 0; c < data[i].getNumberOfColumns(); c++)
				{
					 out[i][n] = data[i].getValue(r, c);
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


	