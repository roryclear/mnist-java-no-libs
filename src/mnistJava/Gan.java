package mnistJava;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

public class Gan {
	static int resolution = 28; //mnist 28*28
	
	static MnistMatrix[] trainData;
	static MnistMatrix[] testData;
	
	static int[][] odTrainData;
	static int[][] odTestData;
	
	static int showAccuracyInterval = 1000;
	
	static int epochs = 100;
	
	public static void main(String[] args) throws IOException
	{		
		//make conjoined Gen and dis like {1,32,64,784,64,32,2} or something
		// flip the dis loss and train gen weights?????/
		
		//discriminator
		Net d = new Net();
		int[] dShape = {784,32,16,2};
		d.layers = dShape;
		d.learningRate = 0.1;
		d.initWeights();
		d.resetNodes();
		
		
		//generator 
		Net g = new Net();
		int[] gShape = {1,16,784};
		g.layers = gShape;
		g.learningRate = 0.1;
		g.initWeights();
		g.resetNodes();
		
		//combined?
		Net c = new Net();
		int[] cShape = {1,16,784,32,16,2};
		c.layers = cShape;
		c.learningRate = 0.1;
		
		
		//data
		trainData = d.readData("mnistdata/train-images.idx3-ubyte","mnistdata/train-labels.idx1-ubyte");
		testData = d.readData("mnistdata/t10k-images.idx3-ubyte","mnistdata/t10k-labels.idx1-ubyte");
		
		odTrainData = d.makeData1D(trainData);
		odTestData = d.makeData1D(testData);
		
		//discriminator epoch 
		double loss = 0;
		double avgLoss = 0;
		double acc = 0;
		int correct = 0;
		for(int i = 0; i < odTrainData.length; i++)
		{
			//get a generated digit
			g.resetNodes();
			Random r = new Random();
			double[] input = {255 * r.nextDouble()};
			g.forward(input, false);
			double[] output = g.getLayer(g.getShape().length - 1);
			int[] disInput = new int[output.length];
			for(int j = 0; j < output.length; j++)
			{
				disInput[j] = (int) output[j] * 255;
			}
			
			//w generated digit
			d.forward(disInput, true);
			
			if(d.getDigit() == 0)
			{
				correct += 1;
			}
			loss += d.getLoss(0);
			
			d.backProp(0);
			d.resetNodes();
			
			//w real digit
			d.forward(odTrainData[i], true);
			
			if(d.getDigit() == 1)
			{
				correct += 1;
			}
			loss += d.getLoss(1);
			
			d.backProp(1);
			d.resetNodes();
			
			avgLoss = (double) loss / (i*2 + 2);
			acc = (double) correct / (i*2 + 2);
			if(i % showAccuracyInterval == 0)
			{
				System.out.println("acc = " + acc + "  avgLoss = " + avgLoss);
	//			System.out.println(correct + " / " + (i*2 + 2));
			}
			
		}
		
		
		//gan test
		for(int i = 0; i < 10; i++)
		{
			Random r = new Random();
			double[] input = {r.nextDouble()};
			System.out.println("input = " + input[0]);
			g.forward(input, false);
			saveOutputToBitmap(g);	
			g.resetNodes();
			try {
				Thread.sleep(30000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
	}
	
	
	
	
	public static void saveOutputToBitmap(Net g) {
		double[] ganOutput = g.getLayer(g.getShape().length - 1); // change
		int[] ganOutputInt = new int[ganOutput.length];
		for(int i = 0; i < ganOutput.length; i++)
		{
			ganOutputInt[i] = (int) (ganOutput[i] * 255);
		}
		BufferedImage newImage = BufferedImage(ganOutputInt);
		saveBMP(newImage,"createdImg.bmp");
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
