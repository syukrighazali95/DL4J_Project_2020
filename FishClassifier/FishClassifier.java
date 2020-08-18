package org.deeplearning4j.examples.DL4JProject.FishClassifier;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class FishClassifier {
    public static Logger log = LoggerFactory.getLogger(FishClassifier.class);
    private static int seed = 1234;

    private static int labelIndex = 1;
    private static int numPossibleLabels = 3;
    private static int channels = 3;
    private static int height = 224;
    private static int width = 224;

    //images are of format given by allowedExtension
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    //random number generator (with seed for reproducibility)
    //private static final Random rng = new Random(123);

    private static String dataDirectory;
    private static String downloadLink;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize = 50;

    //scale input to 0-1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    private static ImageTransform transform;


    public static void main(String[] args) throws Exception{

        //logging process of loading data
        log.info("Loading data");

        //getting home path for fish dataset
        String homePath = System.getProperty("user.home");
        Path dataPath = Paths.get(homePath,".deeplearning4j","data","Fish Species Datasets");
        System.out.println(dataPath);

        Random randNumGen = new Random(seed);
        String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
//        for(String item : allowedExtensions){
//            System.out.println(item);
//        }
//        System.out.println(allowedExtensions[0]);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        FileSplit fileSplit = new FileSplit(new File(dataPath.toString()));

        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelGenerator);
        System.out.println(balancedPathFilter);

        InputSplit[] trainTestSplit = fileSplit.sample(balancedPathFilter, 60,40);
        InputSplit trainData = trainTestSplit[0];
        InputSplit testData = trainTestSplit[1];
        System.out.println(trainData);

        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(height,width, channels, labelMaker);

        trainRecordReader.initialize(trainData);
        testRecordReader.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, labelIndex, numPossibleLabels);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader,batchSize,labelIndex,numPossibleLabels);

        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(0.001))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing/exploding gradient
                .l2(5 * 1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4,4})
                        .name("conv1")
                        .nIn(channels)
                        .nOut(96)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("maxpool1")
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[] {5,5}, new int[] {1,1}, new int[] {2,2})
                        .name("conv2")
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("maxpool2")
                        .kernelSize(3,3)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(128)
                        .activation(Activation.RELU)
                        .dropOut(0.2)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(128)
                        .dropOut(0.2)
                        .build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(224,224,3))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info(model.summary());

//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
//        uiServer.attach(statsStorage);
//
//        model.setListeners(
//            new StatsListener(statsStorage, 5),
//            new ScoreIterationListener(5)
//        );

        model.fit(trainIter, 40);
        Evaluation eval = model.evaluate(testIter);
        log.info(eval.stats());

0
    }
}


