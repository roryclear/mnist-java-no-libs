package mnistJava;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class configurableExample {

	static MnistMatrix[] trainData;  
	static MnistMatrix[] testData; 
	///LAYER SIZES
 	static int inputSize = 28*28;
	static int hiddenSize = 32;
	static int outputSize = 10;
	static double learningRate = 0.1;
	static int epochs = 100;
	static double randomWeightRange = 0.1;

	static int randomSamplesDisplayed = 1;
	
	//one dimensional image data
	static int[][] odTrainData;
	static int[][] odTestData;
	
	
	//save and load weights
	static boolean saveWeights = true;
	static boolean loadWeights = true;
	
	static String saveFile = "confWeights.txt";
	static String loadFile = "confWeights.txt";
	
	static int[] layers = {28*28, 16, 10};
	static ArrayList<Integer[][]> weights = new ArrayList<>();
	static ArrayList<Integer[]> nodes = new ArrayList<>();
	
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

		
	}
	
	
	
	public static void saveWeights()
	{
		//TODO
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
		}
		
	}
	
	public static void forward(int[] data, boolean train)
	{
		//TODO
	}
	
	public static void testNN(MnistMatrix[] trainData, MnistMatrix[] testData)
	{
		//TODO
	}
	
	public static void trainNN(MnistMatrix[] trainData, MnistMatrix[] testData)
	{
		//TODO
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
