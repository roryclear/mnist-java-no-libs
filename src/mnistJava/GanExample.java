package mnistJava;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import javax.imageio.ImageIO;


public class GanExample {
	static MnistMatrix[] trainData;
	static MnistMatrix[] testData;
	
	static MnistMatrix[] trainDataDigit;
	static double[][] odTrainDataDigit;
	
	static double[][] odTrainData;
	static double[][] odTestData;
	
	static int resolution = 28;
	
	static double avgLoss;
	static double accuracy;
	
	static int logInterval = 1000;
	static int epoch = 0;
	
	static double[] epoch0output;
	static double largestChange;
	
	public static void main(String[] args) throws IOException
	{
		int digit = 0;
		
		System.out.println("GAN????");
		int epochs = 100;
		
		//generator
		Net g = new Net();
		int[] gSize = {1,10,784};
		g.learningRate = 0.01;
		g.layers = gSize;
		g.activationFunction = "leakyrelu";
		g.momentum = 0.5;
		g.gradsSize = 0;
		g.resetNodes();
		g.initWeights();
		
		//disriminator
		Net d = new Net();
		int[] dSize = {784,20,2};
		d.learningRate = 0.1;
		d.layers = dSize;
		d.momentum = 0.5;
		d.gradsSize = 0;
		d.activationFunction = "leakyrelu";
		d.resetNodes();
		d.initWeights();
		
		
		//combined
		Net c = new Net();
		int[] cSize = {1,10,784,20,2};
		c.learningRate = 0.1;
		c.momentum = 0.5;
		c.gradsSize = 0;
		c.layers = cSize;
		c.activationFunction = "leakyrelu";
		c.resetNodes();
		c.initWeights();
		
		//data
		trainData = d.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = d.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		
		//make trainData be only chosen digit
		ArrayList<MnistMatrix> digitAl = new ArrayList<>(); 
		for(int i = 0; i < trainData.length; i++)
		{
			if(trainData[i].getLabel() == digit)
			{
			digitAl.add(trainData[i]);
			}
		}
		trainDataDigit = new MnistMatrix[digitAl.size()];
		for(int i = 0; i < digitAl.size(); i++)
		{
			trainDataDigit[i] = digitAl.get(i);
		}
		
		odTrainDataDigit = d.makeData1DDouble(trainDataDigit);
		
		odTrainData = d.makeData1DDouble(trainData);
		odTestData = d.makeData1DDouble(testData);
		
		
		for(epoch = 0; epoch < epochs; epoch++)
		{
		//shuffle
		ArrayList<Integer> shuffled = new ArrayList<>();
		for(int j = 0; j < trainDataDigit.length; j++)
		{
			shuffled.add(j);
		}
		Collections.shuffle(shuffled);
			
		System.out.println("\n\n\n----EPOCH " + epoch +"----\n\n\n\n");
			
		int correct = 0;
		double loss = 0;
		
		System.out.println("\n\n\n-----DISCRIMINATOR-----\n\n\n");
		
		for(int i = 0; i < trainDataDigit.length; i++)
		{
			
			
			int index = shuffled.get(i);
			
			//TRAIN DISCRIMINATOR
			//generated
			g.resetNodes();
			Random r = new Random();
			double[] gInput = {r.nextDouble()};
			g.forward(gInput, false);
			double[] gOutput = g.nodes.get(g.nodes.size() - 1);
			
			d.resetNodes();
			d.forward(gOutput, true);
			d.backProp(0);
			
			if(d.getDigit() == 0)
			{
				correct += 1;
			}
			loss += d.getLoss(0);
			
			//real
			d.resetNodes();
			d.forward(odTrainDataDigit[index], true);
			d.backProp(1);
			if(d.getDigit() == 1)
			{
				correct +=1;
			}
			loss += d.getLoss(1);
			//System.out.println("loss = " + d.getLoss(1));
			if(i % logInterval == 0)
			{
				accuracy = (double) correct / ((i*2) + 2);
				avgLoss = loss / ((i*2) + 2);
				System.out.println(i + " / " + trainDataDigit.length + " acc = " + accuracy + " avgLoss = " + avgLoss);
			}	
			
		}
		
		System.out.println("\n\n\n-----GENERATOR-----\n\n\n");
		
		//TRAIN GENERATOR???
		correct = 0;
		loss = 0;
		for(int i = 0; i < trainDataDigit.length; i++)
		{
			int index = shuffled.get(i);
			//put discriminator weights on combined
			for(int x = 0; x < d.weights.size(); x++)
			{
				c.weights.set((x + g.weights.size()), d.weights.get(x));
			}
			c.resetNodes();
			Random r = new Random();
			double[] cInput = {r.nextDouble()};
			c.forward(cInput, true);
			c.backProp(1);
			if(c.getDigit() == 0)
			{
				correct += 1;
			}
			loss += c.getLoss(0);
			
			
			d.resetNodes();
			d.forward(odTrainDataDigit[index], false);
			if(d.getDigit() == 1)
			{
				correct += 1;
			}
			loss += d.getLoss(1);
			
			
			if(i % logInterval == 0)
			{
				accuracy = (double) correct / ((i*2) + 2);
				avgLoss = loss / ((i*2) + 2);
				System.out.println(i + " / " + trainDataDigit.length + " acc = " + accuracy + " avgLoss = " + avgLoss);
			}	
			
			
		}
		
		//put combined weights on generator
		for(int x = 0; x < g.weights.size(); x++)
		{
			g.weights.set(x, c.weights.get(x));
		}
		
		
		g.resetNodes();
	//	Random r = new Random();
		double[] gInput = {0.5};
		g.forward(gInput, false);
		saveOutputToBitmap(g);
		
		
		if(epoch == 0)
		{
			epoch0output = g.nodes.get(g.nodes.size() - 1);
		}else {
			double[] epochOutput = g.nodes.get(g.nodes.size() - 1);
			largestChange = 0;
			for(int i = 0; i < epochOutput.length - 1; i++)
			{
			//	System.out.println(Math.abs(epochOutput[i] - epoch0output[i]));
				if(Math.abs(epochOutput[i] - epoch0output[i]) > largestChange)
				{
					largestChange = Math.abs(epochOutput[i] - epoch0output[i]);
				}
			}
			System.out.println("largest change = " + largestChange);
		}
		
		
		}//epochs
		
	}
	
	public static int[] rotateAndMirror(int[] image){
		int[] out = new int[image.length];
		int res = (int) Math.sqrt(image.length);
		int[][] image2d = new int[res][res];
		int[][] image2dCopy = new int[res][res];
		
		//copy
		for(int y = 0; y < res; y++)
		{
			for(int x = 0; x < res; x++)
			{
				image2d[y][x] = image[y*28 + x];
				image2dCopy[y][x] = image[y*28 + x];
			}
		}
				
		//rotate
 		for(int y = 0; y < res; y++)
		{
			for(int x = 0; x < res; x++)
			{
				image2d[y][x] = image2dCopy[x][y];
			}
		}
		
		
		//copy back
		for(int y = 0; y < res; y++)
		{
			for(int x = 0; x < res; x++)
			{
				out[y*28 + x] = image2d[y][x];
			}
		}
		
		return out;
	}
	
	public static void saveOutputToBitmap(Net g) {
		double[] ganOutput = g.nodes.get((g.layers.length - 1));
		int[] ganOutputInt = new int[ganOutput.length];
		for(int i = 0; i < ganOutput.length; i++)
		{
			//delete
		//	System.out.println(ganOutput[i]);
			//
			
			ganOutputInt[i] = (int) (ganOutput[i] * 255);
		}
		
		ganOutputInt = rotateAndMirror(ganOutputInt);
		
		BufferedImage newImage = BufferedImage(ganOutputInt);
		saveBMP(newImage,"createdImg" + epoch +  ".bmp");
	}
	
	
    private static void saveBMP( final BufferedImage bi, final String path ){
        try {
            RenderedImage rendImage = bi;
            ImageIO.write(rendImage, "bmp", new File(path));
            //ImageIO.write(rendImage, "PNG", new File(path));
            //ImageIO.write(rendImage, "jpeg", new File(path));
        } catch ( IOException e) {
            e.printStackTrace();
        }
    }
	
	public static BufferedImage BufferedImage(int[] pixels) {
        final BufferedImage res = new BufferedImage(resolution, resolution, BufferedImage.TYPE_INT_RGB );
        for (int x = 0; x < resolution; x++){
            for (int y = 0; y < resolution; y++){
            	int rgb = 65536 * pixels[resolution*x + y] + 256 * pixels[resolution*x + y] + pixels[resolution*x + y];
              //  res.setRGB(x, y, Color.WHITE.getRGB() );
            	res.setRGB(x, y, rgb );
            }
        }
        return res;
	}
	
}
