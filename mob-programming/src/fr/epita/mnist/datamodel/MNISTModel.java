package fr.epita.mnist.datamodel;

import fr.epita.mnist.services.CentroidClassifier;
import fr.epita.mnist.services.MNISTReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class MNISTModel {

    String filePath = "";
    public  List<MNISTImage> images = new ArrayList<>();
    private List<MNISTImage> meanByDigit = new ArrayList<>(10);
    private List<MNISTImage> standardDeviationByDigit = new ArrayList<>(10);
    private double [][] confusionMatrix = new double[28][28];
    private final CentroidClassifier classifier = new CentroidClassifier();

    public List<MNISTImage> getImages() {
        return this.images;
    }

    public MNISTModel(String filePath) {
        try {
            this.filePath=filePath;
            File file = new File(filePath);
            if (file.exists()) {
                MNISTReader mnistReader = new MNISTReader() ;
                this.images  = mnistReader.getAllImages(file);
            }
            else {System.out.print("\nFile Not Found. Please try again.\n");}
        }
        catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        }
    }

    public void TrainCentroidsByAverage() {
        this.meanByDigit = classifier.trainCentroids(this.images);
    }
    public void TrainCentroidsByZ_Score() {
        TrainCentroidsByAverage();
        this.standardDeviationByDigit = classifier.trainStandardDeviation(this.meanByDigit,this.images);
    }

    public void predictUsingCentroids(MNISTImage image) {
        System.out.println("This is most probably the image of the digit number "+(int) classifier.predictWithCentroids(this.meanByDigit,image));
    }
    public void predictUsingZ_Score(MNISTImage image) {
        System.out.println("This is most probably the image of the digit number "+(int) classifier.predictWithZ_Score(this.meanByDigit,this.standardDeviationByDigit,image));
    }

    public void EvaluateByCentroids() {
        this.confusionMatrix = classifier.EvaluateCentroidsModel(this.meanByDigit,this.images);
        System.out.print("\nUsing Centroids\n"+DisplayConfusionMatrixAndModelAccuracy());
    }
    public void EvaluateByCentroidsWithTestImages(List<MNISTImage> images) {
        this.confusionMatrix = classifier.EvaluateCentroidsModel(this.meanByDigit,images);
        System.out.print("\nUsing Centroids\n"+DisplayConfusionMatrixAndModelAccuracy());
    }

    public void EvaluateByZ_Score() {
        this.confusionMatrix = classifier.EvaluateZ_ScoreModel(this.meanByDigit,this.standardDeviationByDigit,this.images);
        System.out.print("\nUsing Z-Score\n"+DisplayConfusionMatrixAndModelAccuracy());
    }
    public void EvaluateByZ_Score_WithTestImages(List<MNISTImage> images) {
        this.confusionMatrix = classifier.EvaluateZ_ScoreModel(this.meanByDigit,this.standardDeviationByDigit,images);
        System.out.print("\nUsing Z-Score\n"+DisplayConfusionMatrixAndModelAccuracy());
    }

    public List<MNISTImage> IsolateImagesOfDigit(int digit, int occurrences){
        return classifier.IsolateN_ImagesOfDigit(this.images,digit,occurrences);
    }

    private String DisplayConfusionMatrixAndModelAccuracy() {
         double validResults=0.0;
         double invalidResults =0.0;
         double maxvalue = GetMatrixMaximum(this.confusionMatrix);
         int maxSpacePerCell=Double.toString((int) maxvalue).length()+2;
         int SpacesLeft;
         StringBuilder toDisplay = new StringBuilder();
         toDisplay.append("-".repeat(maxSpacePerCell * 10)).append("\n");
         for(int index =0; index <confusionMatrix.length* confusionMatrix.length; index++) {
            if (index/10 == index%10) {
                validResults+=this.confusionMatrix[index/10][index%10];
            }
            else{
                invalidResults+=this.confusionMatrix[index/10][index%10];
            }
             SpacesLeft=maxSpacePerCell-Double.toString(this.confusionMatrix[index/10][index%10]).length();
             toDisplay.append("|").append(" ".repeat(SpacesLeft)).append(index / 10 == index % 10 ? "\033[0;1m" : "\033[0;0m").append((int) this.confusionMatrix[index / 10][index % 10]).append(" ");

             if(index%10==9){
                 toDisplay.append("|\n");
                 toDisplay.append("-".repeat(maxSpacePerCell * 10)).append("\n");
             }
         }
         toDisplay.append("Accuracy is % ").append(String.format("%.2f", validResults / (validResults + invalidResults) * 100)).append("\n\n\033[0;0m");
         return toDisplay.toString();
     }

     private double GetMatrixMaximum(double[][] matrix){
        double max =0;
         for(int i =0; i <matrix.length* matrix.length; i++){
             if (matrix[i/10][i%10]>max){
                 max=matrix[i/10][i%10];}
         }
         return max;
     }

}
