package mnistJava;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;


public class GanExample {
	static MnistMatrix[] trainData;
	static MnistMatrix[] testData;
	
	
	static double[][] odTrainData;
	static double[][] odTestData;
	
	static int resolution = 28;
	
	static double avgLoss;
	static double accuracy;
	
	static int logInterval = 10000;
	static int epoch = 0;
	
	static double[] epoch0output;
	
	public static void main(String[] args) throws IOException
	{
		System.out.println("GAN????");
		int epochs = 100;
		
		//generator
		Net g = new Net();
		int[] gSize = {1,10,784};
		g.learningRate = 0.1;
		g.layers = gSize;
		g.activationFunction = "leakyrelu";
		g.momentum = 0.5;
		g.gradsSize = 5;
		g.resetNodes();
		g.initWeights();
		
		//disriminator
		Net d = new Net();
		int[] dSize = {784,10,2};
		d.learningRate = 0.1;
		d.layers = dSize;
		d.momentum = 0.5;
		d.gradsSize = 5;
		d.activationFunction = "leakyrelu";
		d.resetNodes();
		d.initWeights();
		
		
		//combined
		Net c = new Net();
		int[] cSize = {1,10,784,10,2};
		c.learningRate = 0.1;
		c.momentum = 0.5;
		c.gradsSize = 5;
		c.layers = cSize;
		c.activationFunction = "leakyrelu";
		c.resetNodes();
		c.initWeights();
		
		//data
		trainData = d.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = d.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		odTrainData = d.makeData1DDouble(trainData);
		odTestData = d.makeData1DDouble(testData);
		
		for(epoch = 0; epoch < epochs; epoch++)
		{
		
		System.out.println("\n\n\n----EPOCH " + epoch +"----\n\n\n\n");
			
		int correct = 0;
		double loss = 0;
		
		System.out.println("\n\n\n-----DISCRIMINATOR-----\n\n\n");
		
		for(int i = 0; i < trainData.length; i++)
		{
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
			d.forward(odTrainData[i], true);
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
				System.out.println(i + " / " + trainData.length + " acc = " + accuracy + " avgLoss = " + avgLoss);
			}	
			
		}
		
		System.out.println("\n\n\n-----GENERATOR-----\n\n\n");
		
		//TRAIN GENERATOR???
		correct = 0;
		loss = 0;
		for(int i = 0; i < trainData.length; i++)
		{
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
			d.forward(odTrainData[i], false);
			if(d.getDigit() == 1)
			{
				correct += 1;
			}
			loss += d.getLoss(1);
			
			
			if(i % logInterval == 0)
			{
				accuracy = (double) correct / ((i*2) + 2);
				avgLoss = loss / ((i*2) + 2);
				System.out.println(i + " / " + trainData.length + " acc = " + accuracy + " avgLoss = " + avgLoss);
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
			for(int i = 0; i < epochOutput.length - 1; i++)
			{
				System.out.println(Math.abs(epochOutput[i] - epoch0output[i]));
			}
		}
		
		
		}//epochs
		
	}
	
	
	public static void saveOutputToBitmap(Net g) {
		double[] ganOutput = g.getLayer(g.getShape().length - 1); // change
		int[] ganOutputInt = new int[ganOutput.length];
		for(int i = 0; i < ganOutput.length; i++)
		{
			//delete
		//	System.out.println(ganOutput[i]);
			//
			
			ganOutputInt[i] = (int) (ganOutput[i] * 255);
		}
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
