package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class MNISTReader {

    public List<MNISTImage> getAllImages(File file) throws FileNotFoundException {
        List<MNISTImage> mnistImages = new ArrayList<>();
        Scanner scanner = new Scanner(file);
        scanner.nextLine(); // Jump the First Row : Contains Column names
        while(scanner.hasNext()){
            String line = scanner.nextLine();
            MNISTImage image =new MNISTImage();
            image.setLabel(Double.parseDouble(line.substring(0,1)));
            line=line.substring(2);// Jump from label
            image.setPixels(loadLine(line));
            mnistImages.add(image);
        }
        return mnistImages;
    }

    private static double[][] loadLine(String sample) {
        String [] entries  =sample.split(",");
        double [][] entriesAsDouble =new double [28][28];
        for ( int i=0 ; i< entries.length; i++){
            entriesAsDouble[i/28][i%28]=Double.parseDouble(entries[i]);
        }
        return entriesAsDouble;
    }





}
