package mnistJava;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import javax.imageio.ImageIO;

public class configurableExample {

	static MnistMatrix[] trainData;  
	static MnistMatrix[] testData; 
	///LAYER SIZES
 	static int inputSize = 28*28;
	static int hiddenSize = 32;
	static double learningRate = 0.1;
	static int epochs = 100;
	static double randomWeightRange = 0.1;

	
	static int randomSamplesDisplayed = 1;
	static int testNNevery = 10000; //10000
	static int showTrainingAccEvery = 1000;
	
	//one dimensional image data
	static int[][] odTrainData;
	static int[][] odTestData;
	
	
	//save and load weights
	static boolean saveWeights = false;
	static boolean loadWeights = false;
	
	static String saveFile = "confWeights.txt";
	static String loadFile = "confWeights.txt";
	
	static int[] layers = {28*28, 512, 10};
	static ArrayList<double[][]> weights = new ArrayList<>();
	static ArrayList<double[]> nodes = new ArrayList<>();
	
	//may not really need this
	static ArrayList<double[]> nodeInputs = new ArrayList<>();
	
	public static void main(String[] args) throws IOException
	{
		System.out.println("config???");
		
		if(loadWeights)
		{
			loadWeights();
		}else {
			makeRandomWeights();
		}
		//READ data FROM FILES
		trainData = readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		//TEST RANDOM NN
		testNN(trainData, testData);
		
		trainNN(trainData, testData);

		testNN(trainData, testData);
		
	}
	
	
	
	public static void saveWeights()
	{
	    try {
	        FileWriter myWriter = new FileWriter(saveFile);
	        String line = ""+layers[0];
	        for(int i = 1; i < layers.length; i++)
	        {
	        	line+= " " + layers[i];
	        }
	        myWriter.append(line+"\n");
	        
	        for(int i = 0; i < weights.size(); i++)
	        {
	        	for(int y = 0; y < weights.get(i).length; y++)
	        	{
	        		line = ""+ weights.get(i)[y][0];
	        		for(int x = 1; x < weights.get(i)[0].length; x++)
	        		{
	        			line += "," + weights.get(i)[y][x];
	        		}
	        		myWriter.append(line+"\n");
	        	}
	        }
	        
	        
	        myWriter.close();
	        System.out.println("weights saved");
	      } catch (IOException e) {
	        System.out.println("An error occurred.");
	        e.printStackTrace();
	      }
	}
	
	public static void loadWeights()
	{
		//TODO
	}
	
	public static void makeRandomWeights()
	{
		for(int i = 0; i < (layers.length-1); i++)
		{
			double[][] layerWeights = new double[layers[i]][layers[i+1]];
			for(int y = 0; y < layerWeights.length; y++)
			{
				for(int x = 0; x < layerWeights[0].length; x++)
				{
					Random r = new Random();
					double w = -randomWeightRange + 2 * randomWeightRange * r.nextDouble();
					layerWeights[y][x] = w;
				}
			}
			weights.add(layerWeights);
		}
		
	}
	
	public static void forward(int[] data, boolean train)
	{
		for(int i = 0; i < layers[0]; i++)
		{
			nodes.get(0)[i] = sigmoid(data[i]);
		}
		
		for(int i = 1; i < layers.length; i++)
		{
			for(int x = 0; x < layers[i]; x++)
			{
				double total = 0;
				for(int y = 0; y < layers[i-1]; y++)
				{
					total += nodes.get(i - 1)[y] * weights.get(i - 1)[y][x];
				}
				total = sigmoid(total);
				double[] nodesTemp = nodes.get(i);
				nodesTemp[x] = total;
				nodes.set(i, nodesTemp);
				if(train)
				{
					nodeInputs.set(i, nodesTemp);
				}
			}
		}
		
		
	}
	
	
	public static void testNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		//create one dimensional images with values 0 - 255
		int[][] odTestData = makeData1D(testData);
		
		System.out.println("\n ----TEST ON TEST Data------\n");
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
		for(int i = 0; i < layers[layers.length - 1]; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}
		
		
		for(int i = 0; i < testData.length; i++)
		{
			
			resetNodes();
		
			forward(odTestData[i],false);
			int guess = getDigit(nodes.get(nodes.size() - 1));
			hist.replace(guess, hist.get(guess)+1);

			//check if correct
			if(guess == testData[i].getLabel())
			{
				correctHist.replace(guess, correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(nodes.get(nodes.size() - 1),testData[i].getLabel());
			
			
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
		int[] output = new int[layers[layers.length - 1]];
		for(int i = 0; i < layers[layers.length - 1]; i++)
		{
		//	int[] d = bmToArray("mnistdata/" + i + ".bmp"); //comic sans
			int[] d = bmToArray("mnistdata/" + i + "drawn.bmp");
			forward(d,false);
			int guess = getDigit(nodes.get(nodes.size() - 1));
			output[i] = guess;
		}
		
		System.out.println("output for all digits:");
		boolean pass = true; 
		for(int i = 0; i < layers[layers.length - 1]; i++)
		{
			System.out.println(i+": " + output[i]);
			if(output[i] != i)
			{
				pass = false;
			}
		}
		if(pass)
		{
	//		System.exit(0);
		}
		
		
		//TODO
	}
	
	
	public static double sigmoid(double input)
	{
		   double output = 1 / (1 + Math.exp(-input));
		return output;
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
	
	
	public static void resetNodes()
	{
		nodes = new ArrayList<>();
		for(int i = 0; i < layers.length; i++)
		{
			double[] layerNodes = new double[layers[i]];
			nodes.add(layerNodes);
			nodeInputs.add(layerNodes);
		}
	}
	
	public static void trainNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		ArrayList<double[][]> grads = new ArrayList<>();
		for(int i = 0; i < layers.length - 1; i++)
		{
			double[][] gradArray = new double[layers[i]][layers[i+1]];
			grads.add(gradArray);
		}
		
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
		
		for(int i = 0; i < layers[layers.length - 1]; i++)
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
			int guess = getDigit(nodes.get(nodes.size() - 1));
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == trainData[i].getLabel())
			{
				correctHist.replace(guess,correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(nodes.get(nodes.size() - 1),trainData[i].getLabel());
			
			double[] expectedOutput = new double[layers[layers.length - 1]];
			for(int x = 0; x < layers[layers.length - 1]; x++)
			{
				expectedOutput[x] = 0.0;
				if(x==trainData[i].getLabel())
				{
					expectedOutput[x] = 1.0;
				}
			}
		
			grads = new ArrayList<>();
			for(int p = 0; p < layers.length - 1; p++)
			{
				double[][] gradArray = new double[layers[p]][layers[p+1]];
				grads.add(gradArray);
			}
			
			//adjust weights 
			
			//LAST LAYER
			double[][] lastLayerGrads = grads.get(grads.size() - 1);
			for(int y = 0; y < weights.get(weights.size() - 1).length; y++)
			{
				for(int x = 0; x < weights.get(weights.size() - 1)[0].length; x++)
				{
					double output = nodes.get(nodes.size() - 1)[x];
					double expected = expectedOutput[x];
					double dedw = (output - expected)*(output*(1 - output)*(nodes.get(nodes.size() - 2)[y]));
					lastLayerGrads[y][x] = dedw;
				}
			}
			grads.set(grads.size() - 1, lastLayerGrads);
			
			for(int y = 0; y < weights.get(weights.size() - 1).length; y++)
			{
				for(int x = 0; x < weights.get(weights.size() - 1)[y].length; x++)
				{
					double[][] newWeights = weights.get(weights.size() - 1);
					newWeights[y][x] -= learningRate*grads.get(grads.size() - 1)[y][x];
					weights.set(weights.size() - 1, newWeights);
				}
			}
			
			double accuracy = (double) correct/i;
			if(i % showTrainingAccEvery == 0)
			{
			System.out.println("accuracy (training) = " + accuracy);
			}
			
		}//training sample
		
		
		}//epoch
		
		//TODO
	}
	
	
	public static void testNNWithHanddrawn() throws IOException
	{
		///guess hand drawn by me
		int[] drawn = bmToArray("mnistdata/drawn.bmp");
		
		forward(drawn,false);
		
		
		
		//get guess and add to histogram
		int guess = getDigit(nodes.get(nodes.size() - 1));
		
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
		String outs = "0 : "+ nodes.get(nodes.size() - 1)[0];
		for(int i = 1; i < layers[layers.length - 1]; i++)
		{
			outs+=" , " + i + " : " + nodes.get(nodes.size() - 1)[i];
			if(i == 4)
			{
				outs+="\n";
			}
		}
		System.out.println(outs);
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
