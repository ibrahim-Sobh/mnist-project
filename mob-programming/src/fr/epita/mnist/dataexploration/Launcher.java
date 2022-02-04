package fr.epita.mnist.dataexploration;

import fr.epita.mnist.datamodel.MNISTImage;
import fr.epita.mnist.datamodel.MNISTModel;


import java.io.FileNotFoundException;
import java.util.List;
import java.io.*;

public class Launcher {

    public static void main(String[] args) throws FileNotFoundException {

        /* Read MNIS_TrainModel */
        String filePath_train ="/Users/ibrahim/Desktop/AIS/Java/mnist-project/mob-programming/mnist_train.csv";
        MNISTModel model_Train = new MNISTModel(filePath_train);

        /*Display the First image of the Training Model data set */
        System.out.print(model_Train.images.get(0).displayPixels());

        /*Isolate images of a digit from a data set*/
        List <MNISTImage> images_of_Zeros = model_Train.IsolateImagesOfDigit(0,10);
        System.out.print(images_of_Zeros.get(9).displayPixels());

        System.out.print("TEST\n------\n");
        /*Train Centroids by Average and Evaluate the TrainModel*/
        model_Train.TrainCentroidsByAverage();
        model_Train.EvaluateByCentroids();

        /*Train Centroids by Z-score and Evaluate the TrainModel*/
        model_Train.TrainCentroidsByZ_Score();
        model_Train.EvaluateByZ_Score();

        /* TEST MODEL on Trained Model */
        /*-----------------------------*/
        System.out.print("TRAIN\n------\n");
        /* Read MNIS_Test Model */
        String filePath_test ="/Users/ibrahim/Desktop/AIS/Java/mnist-project/mob-programming/mnist_test.csv";
        MNISTModel model_Test = new MNISTModel(filePath_test);

        /*Display the First image of the Testing Model data set */
        System.out.print(model_Test.images.get(0).displayPixels());

        /* Now that we have the model Trained  we can test the TestModel*/
        model_Train.EvaluateByCentroidsWithTestImages(model_Test.getImages());
        model_Train.EvaluateByZ_Score_WithTestImages(model_Test.getImages());

    }
}
