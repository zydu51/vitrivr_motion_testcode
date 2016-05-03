package motion;

import boofcv.abst.feature.describe.ConfigSurfDescribe;
import boofcv.abst.feature.detect.interest.ConfigFast;
import boofcv.abst.feature.detect.interest.ConfigFastHessian;
import boofcv.abst.feature.detect.interest.ConfigGeneralDetector;
import boofcv.abst.feature.orientation.ConfigAverageIntegral;
import boofcv.abst.feature.orientation.ConfigSlidingIntegral;
import boofcv.abst.feature.tracker.PointTracker;
import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.alg.background.BackgroundModelMoving;
import boofcv.alg.distort.PointTransformHomography_F32;
import boofcv.alg.tracker.klt.PkltConfig;
import boofcv.core.image.GConvertImage;
import boofcv.factory.background.ConfigBackgroundBasic;
import boofcv.factory.background.ConfigBackgroundGaussian;
import boofcv.factory.background.FactoryBackgroundModel;
import boofcv.factory.feature.tracker.FactoryPointTracker;
import boofcv.factory.sfm.FactoryMotion2D;
import boofcv.gui.binary.VisualizeBinaryData;
import boofcv.gui.image.ImageGridPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.MediaManager;
import boofcv.io.UtilIO;
import boofcv.io.image.SimpleImageSequence;
import boofcv.io.wrapper.DefaultMediaManager;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;
import georegression.struct.homography.Homography2D_F32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.homography.UtilHomography;

import java.awt.image.BufferedImage;

/**
 * Add different point features for the motion model. 
 * include KLT, SURF, BRIEF and combined KLT and SURF.
 */
public class Motion {
	public static void main(String[] args) {

		// Example with a moving camera.  Highlights why motion estimation is sometimes required
		String fileName = UtilIO.pathExample("1.mjpeg");
		// Camera has a bit of jitter in it.  Static kinda works but motion reduces false positives
//		String fileName = UtilIO.pathExample("background/horse_jitter.mp4");

		// Comment/Uncomment to switch input image type
		ImageType imageType = ImageType.single(GrayF32.class);
//		ImageType imageType = ImageType.il(3, InterleavedF32.class);
//		ImageType imageType = ImageType.il(3, InterleavedU8.class);


		// ---------KLT
		// Configure
		ConfigGeneralDetector confDetector = new ConfigGeneralDetector(300,6,10);
		// Tracker
		PointTracker tracker = FactoryPointTracker.klt(new int[]{1, 2, 4, 8}, confDetector, 3,
				GrayF32.class, null);
		
//		// ---------SURF
//		// Configure
//		ConfigFastHessian configDetector = new ConfigFastHessian();
//		ConfigSurfDescribe.Speed configDescribe = new ConfigSurfDescribe.Speed();
//		ConfigAverageIntegral configOrientation = new ConfigAverageIntegral();
//		// tracker
//		PointTracker tracker = FactoryPointTracker.dda_FH_SURF_Fast(configDetector, configDescribe,
//									configOrientation, GrayF32.class);

//		// ---------BRIEF
//		// Configure
//		ConfigFast configFast = new ConfigFast();
//		ConfigGeneralDetector confDetector = new ConfigGeneralDetector(300,6,10);
//		// tracker
//		PointTracker tracker = FactoryPointTracker.dda_FAST_BRIEF(configFast, confDetector,200,
//									GrayF32.class);
		
//		// ---------Combine SURF and KLT
//		// Configure
//		PkltConfig kltConfig = new PkltConfig();
//		int reactivateThreshold = 50;
//		ConfigFastHessian configDetector = new ConfigFastHessian();
//		ConfigSurfDescribe.Stability configDescribe = new ConfigSurfDescribe.Stability();
//		ConfigSlidingIntegral configOrientation = new ConfigSlidingIntegral();
//		// tracker
//		PointTracker tracker = FactoryPointTracker.combined_FH_SURF_KLT(kltConfig, reactivateThreshold,
//				configDetector,configDescribe,configOrientation,GrayF32.class);

		// This estimates the 2D image motion
		ImageMotion2D<GrayF32,Homography2D_F64> motion2D =
				FactoryMotion2D.createMotion2D(500, 0.5, 3, 100, 0.6, 0.5, false, tracker, new Homography2D_F64());

		ConfigBackgroundBasic configBasic = new ConfigBackgroundBasic(30, 0.005f);

		// Configuration for Gaussian model.  Note that the threshold changes depending on the number of image bands
		// 12 = gray scale and 40 = color
		ConfigBackgroundGaussian configGaussian = new ConfigBackgroundGaussian(20,0.1f);
		configGaussian.initialVariance = 64;
		configGaussian.minimumDifference = 5;

		// Comment/Uncomment to switch background mode
		BackgroundModelMoving background =
//				FactoryBackgroundModel.movingBasic(configBasic, new PointTransformHomography_F32(), imageType);
				FactoryBackgroundModel.movingGaussian(configGaussian, new PointTransformHomography_F32(), imageType);


		MediaManager media = DefaultMediaManager.INSTANCE;
		SimpleImageSequence video =
				media.openVideo(fileName, background.getImageType());
//				media.openCamera(null,640,480,background.getImageType());

		//====== Initialize Images

		// storage for segmented image.  Background = 0, Foreground = 1
		GrayU8 segmented = new GrayU8(video.getNextWidth(),video.getNextHeight());
		// Grey scale image that's the input for motion estimation
		GrayF32 grey = new GrayF32(segmented.width,segmented.height);

		// coordinate frames
		Homography2D_F32 firstToCurrent32 = new Homography2D_F32();
		Homography2D_F32 homeToWorld = new Homography2D_F32();
		homeToWorld.a13 = grey.width/2;
		homeToWorld.a23 = grey.height/2;

		// Create a background image twice the size of the input image.  Tell it that the home is in the center
		background.initialize(grey.width * 2, grey.height * 2, homeToWorld);

		BufferedImage visualized = new BufferedImage(segmented.width,segmented.height,BufferedImage.TYPE_INT_RGB);
		ImageGridPanel gui = new ImageGridPanel(1,2);
		gui.setImages(visualized, visualized);

		ShowImages.showWindow(gui, "Detections", true);

		double fps = 0;
		double alpha = 0.01; // smoothing factor for FPS

		while( video.hasNext() ) {
			ImageBase input = video.next();

			long before = System.nanoTime();
			GConvertImage.convert(input, grey);

			if( !motion2D.process(grey) ) {
				// always run error after processing the whole video.
				// TODO: fix it. 
				throw new RuntimeException("Should handle this scenario");
			}

			Homography2D_F64 firstToCurrent64 = motion2D.getFirstToCurrent();
			UtilHomography.convert(firstToCurrent64, firstToCurrent32);

			background.segment(firstToCurrent32, input, segmented);
			background.updateBackground(firstToCurrent32,input);
			long after = System.nanoTime();

			fps = (1.0-alpha)*fps + alpha*(1.0/((after-before)/1e9));

			VisualizeBinaryData.renderBinary(segmented,false,visualized);
			gui.setImage(0, 0, (BufferedImage)video.getGuiImage());
			gui.setImage(0, 1, visualized);
			gui.repaint();

			System.out.println("FPS = "+fps);

			try {Thread.sleep(5);} catch (InterruptedException e) {}
		}
	}
}