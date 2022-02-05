package fr.epita.mnist.datamodel;

import fr.epita.mnist.services.CentroidClassifier;
import fr.epita.mnist.services.MNISTReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import java.util.stream.Collectors;

public class MNISTModel {

    String filePath = "";
    public  List<MNISTImage> images = new ArrayList<>();
    private List<MNISTImage> meanByDigit = new ArrayList<>(10);
    private List<MNISTImage> standardDeviationByDigit = new ArrayList<>(10);
    private double [][] confusionMatrix = new double[28][28];
    private CentroidClassifier classifier = new CentroidClassifier();

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
        this.standardDeviationByDigit = classifier.trainCentroidsStd(this.meanByDigit,this.images);
    }

    public void predictUsingCentroids(MNISTImage image) {
        System.out.println("This is most probably the image of the digit number "+classifier.predict(this.meanByDigit,image));
    }
    public void predictUsingZ_Score(MNISTImage image) {
        System.out.println("This is most probably the image of the digit number "+classifier.predictStd(this.meanByDigit,this.standardDeviationByDigit,image));
    }

    public void EvaluateByCentroids() {
        this.confusionMatrix = classifier.EvaluateModel(this.meanByDigit,this.images);
        System.out.print("\nUsing Centroids\n"+DisplayConfusionMatrixAndModelAccuracy());
    }
    public void EvaluateByCentroidsWithTestImages(List<MNISTImage> images) {
        this.confusionMatrix = classifier.EvaluateModel(this.meanByDigit,images);
        System.out.print("\nUsing Centroids\n"+DisplayConfusionMatrixAndModelAccuracy());
    }

    public void EvaluateByZ_Score() {
        this.confusionMatrix = classifier.EvaluateModelStd(this.meanByDigit,this.standardDeviationByDigit,this.images);
        System.out.print("\nUsing Z-Score\n"+DisplayConfusionMatrixAndModelAccuracy());
    }
    public void EvaluateByZ_Score_WithTestImages(List<MNISTImage> images) {
        this.confusionMatrix = classifier.EvaluateModelStd(this.meanByDigit,this.standardDeviationByDigit,images);
        System.out.print("\nUsing Z-Score\n"+DisplayConfusionMatrixAndModelAccuracy());
    }

    public List<MNISTImage> IsolateImagesOfDigit(int digit, int occurrences){
        List<MNISTImage> isolated=images.stream().collect(
                Collectors.filtering(mnistImage -> mnistImage.getLabel() == digit,
                        Collectors.toList()));
        if (isolated.stream().count()==0){
            System.out.println("\nNo occurrences were found!\n");
            return new ArrayList<>();
        }
        if (occurrences<=isolated.stream().count())
        {
            return isolated.subList(0,occurrences);
        }
        return isolated;
    }

    private String DisplayConfusionMatrixAndModelAccuracy() {
         double validResults=0.0;
         double invalidResults =0.0;
         double maxvalue = GetMatrixMaximum(this.confusionMatrix);
         int maxSpacePerCell=Double.toString((int) maxvalue).length()+2;
         int SpacesLeft =0;
         String toDisplay ="";
         toDisplay+="-".repeat(maxSpacePerCell*10)+"\n";
         for(int index =0; index <confusionMatrix.length* confusionMatrix.length; index++) {
            if (index/10 == index%10) {
                validResults+=this.confusionMatrix[index/10][index%10];
            }
            else{
                invalidResults+=this.confusionMatrix[index/10][index%10];
            }
             SpacesLeft=maxSpacePerCell-Double.toString(this.confusionMatrix[index/10][index%10]).length();
             toDisplay+= "|"+" ".repeat(SpacesLeft)+ (index/10 == index%10?"\033[0;1m":"\033[0;0m")+(int)this.confusionMatrix[index/10][index%10]+" ";

             if(index%10==9){
                 toDisplay+="|\n";
                 toDisplay+="-".repeat(maxSpacePerCell*10)+"\n";
             }
         }
         toDisplay+="Accuracy is % " + String.format("%.2f", validResults/(validResults+invalidResults)*100)+"\n\n\033[0;0m";
         return toDisplay;
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
