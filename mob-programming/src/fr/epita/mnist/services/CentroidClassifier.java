package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.util.*;
import java.util.stream.Collectors;

public class CentroidClassifier {


    private Map<Double,Long> calculateDistribution(List<MNISTImage> images) {

        Map<Double,Long> distribution= images.stream()
                .collect(Collectors.groupingBy(MNISTImage::getLabel,Collectors.counting()));
        return new TreeMap<>(distribution);
    }

    private List<MNISTImage> IsolateImagesOfDigit(List<MNISTImage> images,int digit)
    {
        return images.stream().collect(
                Collectors.filtering(mnistImage -> mnistImage.getLabel() == digit,
                        Collectors.toList()));

    }
    public  List<MNISTImage> IsolateN_ImagesOfDigit(List<MNISTImage> images, int digit, int occurrences){
        List<MNISTImage> isolated=IsolateImagesOfDigit(images,digit);
        if ((long) isolated.size() ==0){
            System.out.println("\nNo occurrences were found!\n");
            return new ArrayList<>();
        }
        if (occurrences<= (long) isolated.size())
        {
            return isolated.subList(0,occurrences);
        }
        return isolated;
    }

    public List<MNISTImage> trainCentroids (List<MNISTImage> images){
        Map<Double,Long> distributionByDigit = calculateDistribution(images);
        List<MNISTImage>  centroids = new ArrayList<>();
        List<MNISTImage>  IsolatedImages;
        double divider;
        double[][] pixels;


        for (Map.Entry<Double, Long> digit : distributionByDigit.entrySet()){
            MNISTImage centroid =new MNISTImage();
            centroid.setLabel(digit.getKey());
            double [][] centroidMatrix = new double[28][28];
            IsolatedImages =IsolateImagesOfDigit(images,(int) centroid.getLabel());
            for (MNISTImage image : IsolatedImages){
                    divider = digit.getValue();
                    pixels = image.getPixels();
                    for (int j = 0; j < centroidMatrix.length * centroidMatrix.length; j++) {
                        centroidMatrix[j / 28][j % 28] += pixels[j / 28][j % 28] / divider;
                }
            }
            centroid.setPixels(centroidMatrix);
            centroids.add(centroid);
        }
        return centroids;
    }

    public List<MNISTImage> trainStandardDeviation(List<MNISTImage> CentroidsByMean, List<MNISTImage> images){
        Map<Double,Long> distributionByDigit = calculateDistribution(images);
        double[][] pixels;
        double [][] meanMatrix;
        List<MNISTImage>  centroids = new ArrayList<>();
        List<MNISTImage>  IsolatedImages;
        double divider;
        for (Map.Entry<Double, Long> digit : distributionByDigit.entrySet()){
            MNISTImage centroid =new MNISTImage();
            centroid.setLabel(digit.getKey());
            double [][] centroidMatrix = new double[28][28];
            IsolatedImages =IsolateImagesOfDigit(images,(int) centroid.getLabel());
            for (MNISTImage image : IsolatedImages){
                if ((int)image.getLabel()==digit.getKey()) {
                    divider = digit.getValue();
                    pixels= image.getPixels();
                    meanMatrix= CentroidsByMean.get((int) centroid.getLabel()).getPixels();
                    for (int j = 0; j < centroidMatrix.length * centroidMatrix.length; j++) {
                        centroidMatrix [j / 28][j % 28] += ((Math.pow(Math.abs(pixels[j / 28][j % 28]-meanMatrix[j / 28][j % 28]),2))/divider);
                    }
                }
            }
            for (int j = 0; j < centroidMatrix.length * centroidMatrix.length; j++) {
                centroidMatrix[j / 28][j % 28]=Math.sqrt( centroidMatrix[j / 28][j % 28]);
            }
            centroid.setPixels(centroidMatrix);
            centroids.add(centroid);
        }
        return centroids;
    }

    /** This Function uses the Euclidean Distance Formula to measure the distance of each image with the Centroids which is done Pixel by Pixel
     * @param Centroids List of images containing the image of each digit represented in Centroids of Pixel by Pixel
     * @param image image to be predicted
     */
    public double predictWithCentroids(List<MNISTImage> Centroids, MNISTImage image){

        double minimumDistance =256*28*28; // Maximum distance possible :)
        double prediction =10;
        double distance;
        double [][] imageMatrix = image.getPixels();
        double [][] centroidMatrix;

        for (MNISTImage center:  Centroids ) {
            centroidMatrix = center.getPixels();
            distance=0;
            for (int j = 0; j < centroidMatrix.length * centroidMatrix.length; j++) {
               distance+=Math.pow(Math.abs(centroidMatrix[j / 28][j % 28] -imageMatrix[j / 28][j % 28]),2);
            }
            distance=Math.sqrt(distance);
            if (distance<minimumDistance){
                prediction=center.getLabel();
                minimumDistance=distance;
            }
        }
        return prediction;
    }

    /**
     * This Function uses the Z-Score Formula Z= [ point( in our case pixel ) - Mean (of all points in the same set or position ) ] / Standard Deviation(of all points in the same set or position )
     * Then do the SUM(of Z_Score pixel by pixel ) and determines which Digit has the smallest Z-Score in comparison with the image.
     *
     * @param CentroidsMeans List of images containing the image of each digit represented in Centroids/means of Pixel by Pixel
     * @param CentroidsStd List of images containing the image of each digit represented in Std of Pixel by Pixel
     * @param image image to be predicted
     */
    public double predictWithZ_Score(List<MNISTImage> CentroidsMeans, List<MNISTImage> CentroidsStd, MNISTImage image){

        double minimumDistance =256*28*28; // Maximum z_Score possible :)
        double prediction =10;
        double z_Score;
        double [][] imageMatrix = image.getPixels();
        double [][] centroidStdMatrix ;
        double [][] CentroidMeanMatrix ;

        for (MNISTImage center:  CentroidsStd ) {
            CentroidMeanMatrix = CentroidsMeans.get((int)center.getLabel()).getPixels();
            centroidStdMatrix = center.getPixels();
            z_Score=0;
            for (int j = 0; j < centroidStdMatrix.length * centroidStdMatrix.length; j++) {
                if (centroidStdMatrix[j / 28][j % 28] > 0.1) {
                    z_Score += (Math.abs(imageMatrix[j / 28][j % 28] - CentroidMeanMatrix[j / 28][j % 28]) / centroidStdMatrix[j / 28][j % 28]);
                }
                else { // Cast out liar
                    if (imageMatrix[j / 28][j % 28] > 0) {
                        z_Score += 3; /* to the outside of the scope of accepted z_score-Range*/
                    }
                }
            }
            if (z_Score<minimumDistance){
                prediction=center.getLabel();
                minimumDistance=z_Score;
            }
        }
        return prediction;
    }

    public double[][] EvaluateCentroidsModel(List<MNISTImage> Centroids, List<MNISTImage> images){

        double [][] confusionMatrix = new double[10][10];
        for (MNISTImage image:  images ) {
            int label = (int) image.getLabel();
            int prediction = (int) predictWithCentroids(Centroids,image);
            confusionMatrix[label][prediction]+=1;
            }
        return confusionMatrix;
        }

    public double[][] EvaluateZ_ScoreModel(List<MNISTImage> CentroidsMeans, List<MNISTImage> CentroidsStd, List<MNISTImage> images){

        double [][] confusionMatrix = new double[10][10];
        for (MNISTImage image:  images ) {
            int label = (int) image.getLabel();
            int prediction = (int) predictWithZ_Score(CentroidsMeans,CentroidsStd,image);
            confusionMatrix[label][prediction]+=1;
        }
        return confusionMatrix;
    }


}
