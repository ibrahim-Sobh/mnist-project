package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.nio.file.AtomicMoveNotSupportedException;
import java.util.ArrayList;
import java.util.List;

public class CentroidClassifier {

    public List<MNISTImage> trainCentroids (List<MNISTImage> images){
        double [][] distribution = calculateDistribution(images);
        List<MNISTImage>  centroids = new ArrayList<>();
        double divider =0.0;
        double[][] pixels = new double[28][28];
        for (int i=0 ; i< 10; i++){

            MNISTImage centroid =new MNISTImage();
            centroid.setLabel(i);
            double [][] centroidMatrix = new double[28][28];

            for (MNISTImage image : images){
                if ((int)image.getLabel()==i) {
                    divider = distribution[(int) image.getLabel()][1];
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
        double [][] distribution = calculateDistribution(images);
        double[][] pixels =new double[28][28];
        double [][] meanMatrix = new double[28][28];
        List<MNISTImage>  centroids = new ArrayList<>();
        double divider =0.0;
        for (int i=0 ; i< 10; i++){
            MNISTImage centroid =new MNISTImage();
            centroid.setLabel(i);
            double [][] centroidMatrix = new double[28][28];
            for (MNISTImage image : images){
                if ((int)image.getLabel()==i) {
                    divider = distribution[(int) image.getLabel()][1]-1;
                    pixels= image.getPixels();
                    meanMatrix= CentroidsByMean.get(i).getPixels();
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


    public List<MNISTImage> ImagesScaling(List<MNISTImage> OldCentroidsMeans,List<MNISTImage> OldCentroidsStd,List<MNISTImage> images){

        double [][] OdlImageMatrix =  new double [28][28];
        double [][] CentroidsMeanMatrix =new double[28][28];
        double [][] CentroidsStdMatrix = new double[28][28];
        double [][] NewImageMatrix= new double [28][28];
        List<MNISTImage>  Newimages = new ArrayList<>();
        for (MNISTImage image:  images ) {
            MNISTImage newImage = new MNISTImage();
            newImage.setLabel(image.getLabel());
            CentroidsMeanMatrix = OldCentroidsMeans.get((int)image.getLabel()).getPixels();
            CentroidsStdMatrix = OldCentroidsStd.get((int)image.getLabel()).getPixels();
            OdlImageMatrix= image.getPixels();
            double zscore =0.0;
            for (int j = 0; j < NewImageMatrix.length * NewImageMatrix.length; j++) {
                if (OdlImageMatrix[j / 28][j % 28] != 0) {
                    NewImageMatrix[j / 28][j % 28] = (Math.abs(OdlImageMatrix[j / 28][j % 28] - CentroidsMeanMatrix[j / 28][j % 28]) / CentroidsStdMatrix[j / 28][j % 28]);
                    zscore += NewImageMatrix[j / 28][j % 28];
                }

            }
            if (zscore < 3 * 28 * 28) {
                newImage.setPixels(NewImageMatrix);
                Newimages.add(newImage);
            }
        }
        return Newimages;
    }


    private double[][] calculateDistribution(List<MNISTImage> images){
        double[][]  distributions = new double [10][2];
        for (int i=0; i<10;i++) {
            distributions[i][0]=i;
            distributions[i][1]=0;
        }
        for (MNISTImage image : images){
            distributions[(int) image.getLabel()][1]+=1;
        }
        return distributions;
    }

    private List<MNISTImage>  calculateDistributionsMean(List<MNISTImage> images){
        List<MNISTImage>  distributionsMeans = new ArrayList<>();
        double[][]  distributions =calculateDistribution(images);
        double[][] pixels =new double[28][28];
        double divider = 0.0;
        for (int i=0; i<10;i++) {
            MNISTImage mean =new MNISTImage();
            mean.setLabel(i);
            double [][] meanMatrix = new double[28][28];
            for (MNISTImage image : images){
                if ((int)image.getLabel()==i){
                    divider =distributions[i][1];
                    pixels=image.getPixels();
                    for (int j = 0; j < pixels.length * pixels.length; j++) {
                        meanMatrix[j / 28][j % 28] += pixels[j / 28][j % 28] / divider;
                    }
                }
            }
            mean.setPixels((meanMatrix));
            distributionsMeans.add(mean);
        }
        return distributionsMeans;
    }

   public double[][] calculateDistance (List<MNISTImage> Centroids,List<MNISTImage> images){
        double[][]  distances = new double [10][11];
        for (int i=0; i<10;i++) {
           distances[i][0]=i;
        }
        int index =0;
        double distance = 0.0;
       double [][] imageMatrix = new double[28][28];
       double [][] centroidMatrix =new double[28][28];
       for (MNISTImage center:  Centroids ) {
           index=1;
           centroidMatrix = center.getPixels();
           for (MNISTImage image : images) {
               imageMatrix = image.getPixels();
               distance=0;
               for (int j = 0; j < centroidMatrix.length * centroidMatrix.length; j++) {
                   distance+=Math.pow(Math.abs(centroidMatrix[j / 28][j % 28] - imageMatrix[j / 28][j % 28]),2);
               }
               distances[(int) center.getLabel()][index%11]=Math.sqrt(distance);
               index = index + 1;
           }
       }
       return distances;
   }

    public double predict (List<MNISTImage> Centroids,MNISTImage image){

        double minimumDistance =256*28*28; // Maximum distance possible :)
        double prediction =-1;
        double distance =0.0;
        double [][] imageMatrix = image.getPixels();
        double [][] centroidMatrix =new double[28][28];

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

        double minimumDistance =256*28*28; // Maximum distance possible :)
        double prediction =-1;
        double distance =0.0;
        double [][] imageMatrix = image.getPixels();
        double [][] centroidStdMatrix =new double[28][28];
        double [][] CentroidMeanMatrix = new double[28][28];

        for (MNISTImage center:  CentroidsStd ) {
            CentroidMeanMatrix = CentroidsMeans.get((int)center.getLabel()).getPixels();
            centroidStdMatrix = center.getPixels();
            distance=0;
            for (int j = 0; j < centroidStdMatrix.length * centroidStdMatrix.length; j++) {
                if (centroidStdMatrix[j / 28][j % 28] >0.1 && CentroidMeanMatrix[j / 28][j % 28]>0){
                    distance+=(Math.abs(imageMatrix[j / 28][j % 28] -CentroidMeanMatrix[j / 28][j % 28]) /centroidStdMatrix[j / 28][j % 28]);
                }
            }
            if (distance<minimumDistance){
                prediction=center.getLabel();
                minimumDistance=distance;
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
