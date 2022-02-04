package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;


/// fileReader = new FileReader(file);
// read char by char until EOF
// Because it is a huge file we need to read bit by bit


public class MNISTReader {

    public List<MNISTImage> getAllImages(File file) throws FileNotFoundException {
        List<MNISTImage> mnistImages = new ArrayList<>();
        Scanner scanner = new Scanner(file);
        int counter =0;
        scanner.nextLine(); // Jump Column names
        while(scanner.hasNext()){
            String line = scanner.nextLine();
            MNISTImage image =new MNISTImage();
            image.setLabel(Double.valueOf(line.substring(0,1)));
            line=line.substring(2);// Jump from label
            image.setPixels(loadLine(line));
            counter++;
            mnistImages.add(image);
        }
        return mnistImages;
    }

    private static double[][] loadLine(String sample) {

        String [] entries  =sample.split(",");

        double [][] entriesAsDouble =new double [28][28];
        int counter =0;
        for ( int i=0 ; i< entries.length; i++){
            entriesAsDouble[i/28][i%28]=Double.valueOf(entries[i]);
        }
        return entriesAsDouble;
    }





}
