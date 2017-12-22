package test;
import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.FPGrowth;
import weka.associations.FilteredAssociator;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.pmml.consumer.NeuralNetwork;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.Rule;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;
 
public class WekaTest {
	
	private static String data_set_1="/home/jcorvi/workspace/WekaExample/resources/datasets-arie_ben_david/ERA.arff";
	
	private static String data_set_2="/home/jcorvi/workspace/WekaExample/resources/weather.txt";
	
	private static String data_set_hep="/home/jcorvi/workspace/WekaExample/resources/datasets-UCI//UCI/hepatitis.arff";
	
	private static String data_set_diab="/home/jcorvi/workspace/WekaExample/resources/datasets-UCI//UCI/diabetes.arff";
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
	
	/**
	 * 
	 * @param model
	 * @param trainingSet
	 * @param testingSet
	 * @return
	 * @throws Exception
	 */
	public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
	/**
	 * Calcula el porcentaje de aciertos.
	 * @param predictions
	 * @return
	 */
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
	
	/**
	 * Cross Validation Split
	 * @param data
	 * @param numberOfFolds
	 * @return
	 */
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	/**
	 * Salva un clasificador
	 */
	public static void saveClassifier(Classifier cls){
		// serialize model
		 ObjectOutputStream oos;
		try {
			oos = new ObjectOutputStream(new FileOutputStream("/home/jcorvi/workspace/WekaExample/resources/classifiers/"+cls.getClass().getSimpleName()+".model"));
			oos.writeObject(cls);
			oos.flush();
			oos.close();
		
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Salva el conjunto de clasificadores
	 */
	public static void saveClassifiers(Classifier[] clss){
		for (Classifier cls : clss) {
			saveClassifier(cls);
		}
		
	}
	
	
	public  static Instances NumericToNominal(Instances dataProcessed, String variables) throws Exception {
		weka.filters.unsupervised.attribute.NumericToNominal convert = new weka.filters.unsupervised.attribute.NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = variables;
		convert.setOptions(options);
		convert.setInputFormat(dataProcessed);
		Instances filterData = Filter.useFilter(dataProcessed, convert);
		return filterData;
	}
	
	/**
	 * Generacion de Associaciones de la informacion
	 * @param data
	 */
	private static void generateAssociations(Instances data) {
		Apriori apriori = new Apriori();
		apriori.setClassIndex(data.classIndex());
		 
		try {
			Instances data3 = NumericToNominal(data,"1-8");
			apriori.buildAssociations(data3);
			System.out.println(apriori);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile(data_set_diab);
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		
		
		generateAssociations(data);
		
		
		
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 200);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
		
		//Definicion de Modelos
		J48 j48 = new J48();
		j48.setUnpruned(false);
		J48 j48_unpruned = new J48();
		j48_unpruned.setUnpruned(true);
		
		/*
		 * Aplicarle filtros a los datos
		FilteredClassifier fc = new FilteredClassifier();
		fc.setClassifier(j48);
		fc.buildClassifier(data);
		*/
		// build associator
		
		/*Apriori apriori = new Apriori();
		apriori.setClassIndex(data.classIndex());
		apriori.buildAssociations(data);
		// output associator
		System.out.println(apriori);
		*/
		/*
		Apriori fa = new Apriori();
		NumericToNominal numTo = new NumericToNominal();
		
		fa.buildAssociations(data);
		AssociationRules aRules =fa.getAssociationRules();
		for (AssociationRule rule : aRules.getRules()) {
			System.out.println(rule.toString());
		}
		
		
		*/
		/*
		FilteredAssociator filteredAssociator = new FilteredAssociator();
		filteredAssociator.setClassIndex(data.classIndex());
		filteredAssociator.buildAssociations(data);
		System.out.println(filteredAssociator);
		*/
		
		/*
		FPGrowth fPGrowth = new FPGrowth();
		fPGrowth.buildAssociations(data);
		System.out.println(fPGrowth);
		*/
		//NeuralNetwork neuralNetwork = new NeuralNetwork()
		BayesNet bayesNet = new BayesNet();
		// Use a set of classifiers
		Classifier[] models = { 
				j48_unpruned, // a decision tree,
				j48,
				bayesNet,
				new PART(), 
				new DecisionTable(),//decision table majority classifier
				new DecisionStump() //one-level decision tree
				
		};
 
		// Run for each model 
		for (int j = 0; j < models.length; j++) {
 
			// Collect every group of predictions for current model in a FastVector
			FastVector predictions = new FastVector();
 
			// For each training-testing split pair, train and test the classifier
			for (int i = 0; i < trainingSplits.length; i++) {
				
				
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
				predictions.appendElements(validation.predictions());
				//plot_roc_curve(validation);
				
				// Uncomment to see the summary for each training-testing pair.
				//System.out.println(models[j].toString());
			}
			
			plot_roc_curve(predictions);
			
			// Calculate overall accuracy of current classifier on all splits
			double accuracy = calculateAccuracy(predictions);
 
			// Print current classifier's name and accuracy in a complicated,
			// but nice-looking way.
			System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy)
					+ "\n---------------------------------");
			
			saveClassifier(models[j]);
			
			
			
		}
 
	}


	/**
	    * takes one argument: dataset in ARFF format (expects class to
	    * be last attribute)
	    */
	   public static void plot_roc_curve(FastVector predictions) throws Exception {
	     
	 
	     // generate curve
	     ThresholdCurve tc = new ThresholdCurve();
	     int classIndex = 0;
	     Instances result = tc.getCurve(predictions, classIndex);
	 
	     // plot curve
	     ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
	     vmc.setROCString("(Area under ROC = " +
	         Utils.doubleToString(tc.getROCArea(result), 4) + ")");
	     vmc.setName(result.relationName());
	     PlotData2D tempd = new PlotData2D(result);
	     tempd.setPlotName(result.relationName());
	     tempd.addInstanceNumberAttribute();
	     // specify which points are connected
	     boolean[] cp = new boolean[result.numInstances()];
	     for (int n = 1; n < cp.length; n++)
	       cp[n] = true;
	     tempd.setConnectPoints(cp);
	     // add plot
	     vmc.addPlot(tempd);
	 
	     // display curve
	     String plotName = vmc.getName();
	     final javax.swing.JFrame jf =
	       new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
	     jf.setSize(500,400);
	     jf.getContentPane().setLayout(new BorderLayout());
	     jf.getContentPane().add(vmc, BorderLayout.CENTER);
	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	       public void windowClosing(java.awt.event.WindowEvent e) {
	       jf.dispose();
	       }
	     });
	     jf.setVisible(true);
	   }
	
	
}