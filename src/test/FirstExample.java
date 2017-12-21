package test;
import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
public class FirstExample {

	public static void main(String[] args) {
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader("/home/jcorvi/workspace/WekaExample/resources/train.g"));
			Instances train = new Instances(reader);
			reader.close();
			
			
			reader = new BufferedReader(new FileReader("/home/jcorvi/workspace/WekaExample/resources/pred.g"));
			Instances pred = new Instances(reader);
			reader.close();
			
			// setting class attribute
			train.setClassIndex(train.numAttributes() - 1);
			
			//Remove rm = new Remove();
			//rm.setAttributeIndices("1");  // remove 1st attribute
			// classifier
			J48 j48 = new J48();
			j48.setUnpruned(true);        // using an unpruned J48
			// meta-classifier
			FilteredClassifier fc = new FilteredClassifier();
			//fc.setFilter(rm);
			fc.setClassifier(j48);
			// train and make predictions
			fc.buildClassifier(train);
			for (int i = 0; i < train.numInstances(); i++) {
			   double pred_value = fc.classifyInstance(train.instance(i));
			   System.out.print("ID: " + train.instance(i).value(0));
			   System.out.print(", actual: " + train.classAttribute().value((int) train.instance(i).classValue()));
			   System.out.println(", predicted: " + train.classAttribute().value((int) pred_value));
			}
			
			
			
			
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 

	}

}
