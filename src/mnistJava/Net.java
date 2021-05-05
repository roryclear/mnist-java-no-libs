package mnistJava;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Net {
	static MnistMatrix[] trainData;  
	static MnistMatrix[] testData; 
	///LAYER SIZES
	static double learningRate = 0.1;
	static int epochs = 100; //100
	static double randomWeightRange = 0.1; 
	
	static double momentum = 0.5;
	static int gradsSize = 5;
	
	static int randomSamplesDisplayed = 1;
	static int testNNevery = 10000; //10000
	static int showTrainingAccEvery = 1000; //1000
	
	//conf
	static int layers[] = {784,32,10};
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
	
	
	static String saveFile = "confWeights";
	static String loadFile = "confWeights";
	
	
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
	
	public static double sigmoid(double input)
	{
		double output = 1 / (1 + Math.exp(-input));
		return output;
	}
	
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
	public static int getDigit()
	{
		int output = 0;
		double[] outputs = nodes.get(nodes.size() - 1);
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
    
	public void initWeights()
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
	
	public void resetNodes()
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
	
}
