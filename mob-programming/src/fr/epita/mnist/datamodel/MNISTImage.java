package fr.epita.mnist.datamodel;

public class MNISTImage {

     double label;

     double[][] pixels= new double[28][28];

     public double getLabel() {
          return label;
     }

     public double[][] getPixels() {
          return pixels;
     }

     public void setLabel(double label) {
          this.label = label;
     }

     public void setPixels(double[][] pixels) {
          this.pixels = pixels;
     }


     public String displayPixels ( ){
          double [][] matrix = this.pixels;
          int newLineCounter =0;
          String image = "";
          for ( int i=0 ; i< matrix.length*matrix.length; i++){

               if (newLineCounter <i/28) {
                    image+="\n";
                    newLineCounter+=1;
               }
               double  value = matrix[i / 28][i % 28];
               if (value>128) {
                    image+="XX";
               }
               else
               {image+="..";
               }
          }
          image+="\n\n";
          return image;


     }



}
