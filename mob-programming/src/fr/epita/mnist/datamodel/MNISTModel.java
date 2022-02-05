package fr.epita.mnist.datamodel;

import fr.epita.mnist.services.CentroidClassifier;
import fr.epita.mnist.services.MNISTReader;
import org.w3c.dom.ls.LSOutput;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class MNISTModel {
    String filePath = "";
    public List<MNISTImage> images = new ArrayList<>();
    private List<MNISTImage> meanByDigit = new ArrayList<>(10);
    private List<MNISTImage> standardDeviationByDigit = new ArrayList<>(10);
    private double [][] confusionMatrix = new double[28][28];
    private CentroidClassifier classifier = new CentroidClassifier();

    public List<MNISTImage> getImages() {
        return this.images;
    }

    public MNISTModel(String filePath) throws FileNotFoundException {
        this.filePath=filePath;
        File file = new File(filePath);
        if (file.exists()) {
            MNISTReader mnistReader = new MNISTReader() ;
            this.images  = mnistReader.getAllImages(file);
        }
        else {System.out.print("file not found");
        }
    }

    public void TrainCentroidsByAverage() {
        this.meanByDigit = classifier.trainCentroids(this.images);
    }
    public void TrainCentroidsByZ_Score() {
        TrainCentroidsByAverage();
        this.standardDeviationByDigit = classifier.trainCentroidsStd(this.meanByDigit,this.images);
    }

    public void EvaluateByCentroids() {
        this.confusionMatrix = classifier.EvaluateModel(this.meanByDigit,this.images);
        System.out.print(DisplayConfusionMatrix());
    }
    public void EvaluateByCentroidsWithTestImages(List<MNISTImage> images) {
        this.confusionMatrix = classifier.EvaluateModel(this.meanByDigit,images);
        System.out.print(DisplayConfusionMatrix());
    }

    public void EvaluateByZ_Score() {
        this.confusionMatrix = classifier.EvaluateModelStd(this.meanByDigit,this.standardDeviationByDigit,this.images);
        System.out.print(DisplayConfusionMatrix());
    }
    public void EvaluateByZ_Score_WithTestImages(List<MNISTImage> images) {
        this.confusionMatrix = classifier.EvaluateModelStd(this.meanByDigit,this.standardDeviationByDigit,images);
        System.out.print(DisplayConfusionMatrix());
    }

    public List<MNISTImage> IsolateImagesOfDigit(int digit, int occurrences){
        List<MNISTImage> isolated=images.stream().collect(
                Collectors.filtering(mnistImage -> mnistImage.getLabel() == digit,
                        Collectors.toList()));
        if (isolated.stream().count()==0){
            System.out.println("No occurrences were found!\n");
            return new ArrayList<>();
        }
        if (occurrences<=isolated.stream().count())
        {
            return isolated.subList(0,occurrences);
        }
        return isolated;
    }

    private String DisplayConfusionMatrix () {
         double validResults=0.0;
         double invalidResults =0.0;
         String toDisplay ="";
         for(int i =0; i <10; i++){
             toDisplay+="-------------------------------------------------------------------\n";
             for(int j=0 ; j <10; j++)
             {   if (i==j) {
                 validResults+=this.confusionMatrix[i][j];}
                 else{
                     invalidResults+=this.confusionMatrix[i][j];
                 }
                 toDisplay+= confusionMatrix[i][j]+ " | ";
             }
             toDisplay+="\n";
         }
         toDisplay+="-------------------------------------------------------------------\n\n";
         toDisplay+="Accuracy is % " + String.format("%.2f", validResults/(validResults+invalidResults)*100)+"\n\n";

         return toDisplay;
     }

}
