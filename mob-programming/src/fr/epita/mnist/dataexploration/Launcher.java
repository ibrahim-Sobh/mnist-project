package fr.epita.mnist.dataexploration;

import fr.epita.mnist.datamodel.MNISTImage;
import fr.epita.mnist.datamodel.MNISTModel;


import java.io.FileNotFoundException;
import java.util.List;
import java.io.*;

public class Launcher {

    public static void main(String[] args) throws FileNotFoundException {

        /* Read MNIS_TrainModel */
        String filePath_train ="/Users/ibrahim/IdeaProjects/mnist_train.csv";
        MNISTModel model_Train = new MNISTModel(filePath_train);


        /*Isolate images of a digit from a data set*/
        List <MNISTImage> images_of_Zeros = model_Train.IsolateImagesOfDigit(0,11);
        System.out.print(images_of_Zeros.get(9).displayPixels());

        System.out.print("\nTRAIN\n------\n");
        /*Train Centroids by Average and Evaluate the TrainModel*/
        model_Train.TrainCentroidsByAverage();
        model_Train.EvaluateByCentroids();

        /*Train Centroids by Z-score and Evaluate the TrainModel*/
        model_Train.TrainCentroidsByZ_Score();
        model_Train.EvaluateByZ_Score();

        /* TEST MODEL using Trained Model */
        /*-----------------------------*/
        /* Read MNIS_Test Model */
        String filePath_test ="/Users/ibrahim/IdeaProjects/mnist_test.csv";
        MNISTModel model_Test = new MNISTModel(filePath_test);

        /*Display & Predict an image of the Testing Model data using Centroids */
        System.out.print(model_Test.images.get(0).displayPixels());
        model_Train.predictUsingCentroids(model_Test.images.get(0));

        /*Display & Predict another image of the Training Model data set using Z-Score */
        System.out.print(model_Test.images.get(4).displayPixels());
        model_Train.predictUsingZ_Score(model_Test.images.get(4));

        System.out.print("\nTEST\n------\n");

        /* Now that we have the model Trained  we can test the TestModel*/
        model_Train.EvaluateByCentroidsWithTestImages(model_Test.getImages());
        model_Train.EvaluateByZ_Score_WithTestImages(model_Test.getImages());

    }
}
