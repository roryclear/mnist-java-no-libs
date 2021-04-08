package mnistJava;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;

import javax.imageio.ImageIO;


public class configurableExampleWithMomentum {
	static MnistMatrix[] trainData;  
	static MnistMatrix[] testData; 
	///LAYER SIZES
	static double learningRate = 0.1;
	static int epochs = 1; //100
	static double randomWeightRange = 0.1; 
	
	static double momentum = 0.5;
	static int gradsSize = 1;
	
	static int randomSamplesDisplayed = 1;
	static int testNNevery = 10000; //10000
	static int showTrainingAccEvery = 1000; //1000
	
	//conf
	static int layers[] = {784,16,10};
	static int outputSize = layers[layers.length - 1];
	static int numberOfLayers = layers.length;
	
	
	//init nodes
	static ArrayList<double[]> nodes = new ArrayList<>();
	
	//totals for bp
	static ArrayList<double[]> nodesTotal = new ArrayList<>();
	
	//weights
	static ArrayList<double[][]> weights = new ArrayList<>();
	
	//grads + prev grads
	//static ArrayList<double[][]> grads = new ArrayList<>();	
	static ArrayList<ArrayList<double[][]>> grads = new ArrayList<>();	
	
	//one dimensional data
	static int[][] odTrainData;
	static int[][] odTestData;
	
	//save and load weights
	static boolean saveWeights = false;
	static boolean loadWeights = false;
	
	static String saveFile = "confWeights";
	static String loadFile = "confWeights";
	
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
		
		for(int x = 0; x < layers[0]; x++)
		{
			nodes.get(0)[x] = sigmoid(data[x]);
			if(train)
			{
				nodesTotal.get(0)[x] += data[x];
			}
		}
		
		//get rest
		for(int i = 1; i < layers.length; i++)
		{
			for(int x = 0; x < layers[i]; x++)
			{
				double total = 0;
				for(int y = 0; y < layers[i-1]; y++)
				{
					total += nodes.get(i-1)[y]*weights.get(i-1)[y][x];
				}
				total = sigmoid(total);
				nodes.get(i)[x] = total;
				if(train)
				{
					nodesTotal.get(i)[x] += total;
				}
			}
		}
		
	
		
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
		for(int i = 0; i < layers[numberOfLayers - 1]; i++)
		{
			hist.put(i, 0);
			correctHist.put(i,0);
		}
		
		for(int i = 0; i < testData.length; i++)
		{
			resetNodes();
		
			
			forward(odTestData[i],false);
			
			
			//get guess and add to histogram
			int guess = getDigit(nodes.get(numberOfLayers - 1));
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == testData[i].getLabel())
			{
				correctHist.replace(guess, correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(nodes.get(numberOfLayers - 1),testData[i].getLabel());
			
			
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
		int[] output = new int[layers[numberOfLayers - 1]];
		for(int i = 0; i < layers[numberOfLayers - 1]; i++)
		{
		//	int[] d = bmToArray("mnistdata/" + i + ".bmp"); //comic sans
			int[] d = bmToArray("mnistdata/" + i + "drawn.bmp");
			forward(d,false);
			int guess = getDigit(nodes.get(numberOfLayers - 1));
			output[i] = guess;
		}
		
		System.out.println("output for all digits:");
		boolean pass = true; 
		for(int i = 0; i < layers[numberOfLayers - 1]; i++)
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
		

		
		
		
	}
	
	public static void makeRandomWeights()
	{		
		for(int i = 0; i < layers.length - 1; i++)
		{
			double[][] layerWeights = new double[layers[i]][layers[i+1]];
			for(int y = 0; y < layerWeights.length; y++)
			{
				for(int x = 0; x < layerWeights[y].length; x++)
				{
					Random r = new Random();
					double w = -randomWeightRange + 2 * randomWeightRange * r.nextDouble();
					layerWeights[y][x] = w;
				}
			}
			
			weights.add(layerWeights);
		}
		
	}
	
	public static void loadWeights()
	{
		String layersString = ""+layers[0];
		for(int i = 1; i < layers.length; i++)
		{
			layersString+="-" + layers[i];
		}
		for(int i = 0; i < layers.length - 1; i++)
		{
			double[][] layerWeights = new double[layers[i]][layers[i+1]];		
			weights.add(layerWeights);
		}
		
	    try {
	        File myObj = new File(loadFile+layersString+".txt");
	        Scanner myReader = new Scanner(myObj);
	        myReader.nextLine();
	       
	        int layer = 0;
	        while (myReader.hasNextLine()) 
	        {	        
	        int y = 0;
	        while(y < weights.get(layer).length)
	        {
	          String data = myReader.nextLine();
	          String[] weightStrings = data.split(",");
	          if(y < weights.get(layer).length)
	          {
	        	  for(int x = 0; x < weightStrings.length; x++)
	        	  {
	        		  weights.get(layer)[y][x] = Double.parseDouble(weightStrings[x]);
	        	  }
	          }
	          y++;
	         }
	        layer++;
	        y = 0;
	        }
	        myReader.close();
	      } catch (FileNotFoundException e) {
	        System.out.println("An error occurred.");
	        e.printStackTrace();
	      }
	}
	
	public static void saveWeights()
	{
		String layersString = ""+layers[0];
		for(int i = 1; i < layers.length; i++)
		{
			layersString+="-" + layers[i];
		}
	    try {
	        FileWriter myWriter = new FileWriter(saveFile+layersString+".txt");
	        myWriter.append(layersString);
	        
	        for(int i = 0; i < weights.size(); i++)
	        {
	        
	        for(int y = 0; y < weights.get(i).length; y++)
	        {
	        	String line = ""+weights.get(i)[y][0];
	        	for(int x = 1; x < weights.get(i)[y].length; x++)
	        	{
	        		line = line + "," + weights.get(i)[y][x];
	        	}
	        	myWriter.append("\n" + line);
	        }
	        
	        }
	        
	        myWriter.close();
	        System.out.println("weights saved");
	      } catch (IOException e) {
	        System.out.println("An error occurred.");
	        e.printStackTrace();
	      }
	}
	
	public static void setGrads()
	{
		if(grads.size() < gradsSize)
		{
			for(int i = 0; i < gradsSize; i++)
			{
				ArrayList<double[][]> layerGrads = new ArrayList<>();
				grads.add(layerGrads);
			}
			
		}else {
			for(int i = 0; i < gradsSize - 1; i++)
			{
				grads.set(i, grads.get(i+1));
			}
			ArrayList<double[][]> layerGrads = new ArrayList<>();
			grads.set(gradsSize - 1,  layerGrads);
		}
	}
	
	public static void resetNodes()
	{
		nodes = new ArrayList<>();
		nodesTotal = new ArrayList<>();
		for(int i = 0; i < layers.length; i++)
		{
			double[] layerNodes = new double[layers[i]];
			double[] layerNodesTotal = new double[layers[i]];
			nodes.add(layerNodes);
			nodesTotal.add(layerNodesTotal);
		}
		
	}
	
	public static void trainNN(MnistMatrix[] trainData, MnistMatrix[] testData) throws IOException
	{
		

		
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
		for(int i = 0; i < layers[numberOfLayers - 1]; i++)
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
			}

			
			forward(odTrainData[i],true);
			
			
			//get guess and add to histogram
			int guess = getDigit(nodes.get(numberOfLayers - 1));
			hist.replace(guess, hist.get(guess)+1);
			
			//check if correct
			if(guess == trainData[i].getLabel())
			{
				correctHist.replace(guess,correctHist.get(guess)+1);
				correct+=1;
			}
			
			//getLoss
			loss+=getLoss(nodes.get(numberOfLayers - 1),trainData[i].getLabel());
			
			double[] expectedOutput = new double[layers[numberOfLayers - 1]];
			for(int x = 0; x < layers[numberOfLayers - 1]; x++)
			{
				expectedOutput[x] = 0.0;
				if(x==trainData[i].getLabel())
				{
					expectedOutput[x] = 1.0;
				}
			}
			
			if(gradsSize > 1)
			{
			setGrads();
			ArrayList<double[][]> currentGrads = new ArrayList<>();	
			for(int r = 0; r < layers.length - 1; r++)
			{
				double[][] layerGrads = new double[weights.get(r).length][weights.get(r)[0].length];
				currentGrads.add(layerGrads);
			} 
			
			grads.set(gradsSize - 1,  currentGrads);
			
			//output
			for(int y = 0; y < weights.get(numberOfLayers - 2).length; y++)
			{
				double L1Output = nodesTotal.get(numberOfLayers - 2)[y];
				for(int x = 0; x < weights.get(numberOfLayers - 2)[y].length; x++)
				{
					double output = nodesTotal.get(numberOfLayers - 1)[x];
					double expected = expectedOutput[x];
					double dedw = (output - expected)*(output*(1 - output)*(L1Output));
					grads.get(gradsSize - 1).get(numberOfLayers - 2)[y][x] += dedw;
				}
			}
			
			
			
			//all other weights
			for(int r = numberOfLayers - 3; r > -1; r--)
			{
			
			for(int y = 0; y < weights.get(r).length; y++)
			{
				for(int x = 0; x < weights.get(r)[y].length; x++)
				{
					double totalError = 0;
					for(int n = 0; n < nodesTotal.get(r+2).length; n++)	
					{
						totalError += (weights.get(r+1)[x][n]*grads.get(gradsSize - 1).get(r+1)[x][n])/nodes.get(r+2).length;
					}
					totalError = totalError*(nodesTotal.get(r+1)[x]*(1 - nodesTotal.get(r+1)[x])*nodesTotal.get(r)[y]);
					
					
					grads.get(gradsSize - 1).get(r)[y][x] += totalError;
				}
			}
			
			}
			
			for(int r = 0; r < numberOfLayers - 1; r++)
			{
				
			for(int y = 0; y < weights.get(r).length; y++)
			{
				for(int x = 0; x < weights.get(r)[y].length; x++)
				{
					double gradient = 0;
					for(int g = 0; g < gradsSize; g++)
					{
						if(grads.get(gradsSize - 1 - g).size() > 0)
						{
						gradient += grads.get(gradsSize - 1 - g).get(r)[y][x]*(Math.pow(momentum, g));
						}
					}
					gradient = gradient/gradsSize;
					
					weights.get(r)[y][x] -= (learningRate*gradient);
				}
			}
			}
			}else {	//gradsSize < 1
				ArrayList<double[][]> grads = new ArrayList<>();	
				for(int r = 0; r < layers.length - 1; r++)
				{
					double[][] layerGrads = new double[weights.get(r).length][weights.get(r)[0].length];
					grads.add(layerGrads);
				} 

				
				//output
				for(int y = 0; y < weights.get(numberOfLayers - 2).length; y++)
				{
					double L1Output = nodesTotal.get(numberOfLayers - 2)[y];
					for(int x = 0; x < weights.get(numberOfLayers - 2)[y].length; x++)
					{
						double output = nodesTotal.get(numberOfLayers - 1)[x];
						double expected = expectedOutput[x];
						double dedw = (output - expected)*(output*(1 - output)*(L1Output));
						grads.get(numberOfLayers - 2)[y][x] += dedw;
					}
				}
				
				
				
				//all other weights
				for(int r = numberOfLayers - 3; r > -1; r--)
				{
				
				for(int y = 0; y < weights.get(r).length; y++)
				{
					for(int x = 0; x < weights.get(r)[y].length; x++)
					{
						double totalError = 0;
						for(int n = 0; n < nodesTotal.get(r+2).length; n++)	
						{
							totalError += (weights.get(r+1)[x][n]*grads.get(r+1)[x][n])/nodes.get(r+2).length;
						}
						totalError = totalError*(nodesTotal.get(r+1)[x]*(1 - nodesTotal.get(r+1)[x])*nodesTotal.get(r)[y]);
						
						
						grads.get(r)[y][x] += totalError;
					}
				}
				
				}
				
				for(int r = 0; r < numberOfLayers - 1; r++)
				{
				
				for(int y = 0; y < weights.get(r).length; y++)
				{
					for(int x = 0; x < weights.get(r)[y].length; x++)
					{
						weights.get(r)[y][x] -= (learningRate*grads.get(r)[y][x]);
					}
				}
				}
			}
	
							
			DecimalFormat df = new DecimalFormat("#.####");
			if(i % showTrainingAccEvery == 0)
			{
			//remove
			double accuracy = (double) correct/i;
			double avgLoss = loss/i;
			System.out.println("accuracy (training) = " + df.format(accuracy) + "	loss (training) = " + df.format(avgLoss));
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
			int guess = getDigit(nodes.get(numberOfLayers - 1));
		
		
			System.out.println("\n----YOU DREW A " + guess + "?------- ");
			String outs = "0 : " + nodes.get(numberOfLayers - 1)[0];
			for(int i = 1; i < layers[numberOfLayers - 1]; i++)
			{
				outs+=" , " + i + " : " + nodes.get(numberOfLayers - 1)[i];
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


	