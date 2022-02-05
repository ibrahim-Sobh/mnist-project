package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.security.PublicKey;
import java.util.*;
import java.util.stream.Collectors;
;

public class CentroidClassifier {


    private Map<Double,Long> calculateDistribution(List<MNISTImage> images) {

        Map<Double,Long> distribution= images.stream()
                .collect(Collectors.groupingBy(MNISTImage::getLabel,Collectors.counting()));
        TreeMap<Double, Long> sortedDistribution = new TreeMap<>();
        sortedDistribution.putAll(distribution);
        return sortedDistribution;
    }

    public List<MNISTImage> trainCentroids (List<MNISTImage> images){
        Map<Double,Long> distributionByDigit = calculateDistribution(images);
        List<MNISTImage>  centroids = new ArrayList<>();
        double divider =0.0;
        double[][] pixels = new double[28][28];

        for (Map.Entry<Double, Long> digit : distributionByDigit.entrySet()){
            MNISTImage centroid =new MNISTImage();
            centroid.setLabel(digit.getKey());
            double [][] centroidMatrix = new double[28][28];

            for (MNISTImage image : images){
                if ((int)image.getLabel()==digit.getKey()) {
                    divider = digit.getValue();
                    pixels = image.getPixels();
                    for (int j = 0; j < centroidMatrix.length * centroidMatrix.length; j++) {
                        centroidMatrix[j / 28][j % 28] += pixels[j / 28][j % 28] / divider;
                    }
                }
            }
            centroid.setPixels(centroidMatrix);
            centroids.add(centroid);
        }
        return centroids;
    }

    public List<MNISTImage> trainCentroidsStd (List<MNISTImage> CentroidsByMean,List<MNISTImage> images){
        Map<Double,Long> distributionByDigit = calculateDistribution(images);
        double[][] pixels =new double[28][28];
        double [][] meanMatrix = new double[28][28];
        List<MNISTImage>  centroids = new ArrayList<>();
        double divider =0.0;
        for (Map.Entry<Double, Long> digit : distributionByDigit.entrySet()){
            MNISTImage centroid =new MNISTImage();
            centroid.setLabel(digit.getKey());
            double [][] centroidMatrix = new double[28][28];
            for (MNISTImage image : images){
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


    public double predict (List<MNISTImage> Centroids,MNISTImage image){

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

    public double predictStd(List<MNISTImage> CentroidsMeans,List<MNISTImage> CentroidsStd,MNISTImage image){

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
                        z_Score += 3;
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


    public double[][] EvaluateModel (List<MNISTImage> Centroids,List<MNISTImage> images){

        double [][] confusionMatrix = new double[10][10];
        for (MNISTImage image:  images ) {
            int label = (int) image.getLabel();
            int prediction = (int) predict(Centroids,image);
            confusionMatrix[label][prediction]+=1;
            }
        return confusionMatrix;
        }

    public double[][] EvaluateModelStd (List<MNISTImage> CentroidsMeans,List<MNISTImage> CentroidsStd,List<MNISTImage> images){

        double [][] confusionMatrix = new double[10][10];
        for (MNISTImage image:  images ) {
            int label = (int) image.getLabel();
            int prediction = (int) predictStd(CentroidsMeans,CentroidsStd,image);
            confusionMatrix[label][prediction]+=1;
        }
        return confusionMatrix;
    }



}
