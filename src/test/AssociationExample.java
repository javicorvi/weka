package test;

import java.io.BufferedReader;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.filters.Filter;

public class AssociationExample {
	
	private static String data_set_1="/home/jcorvi/workspace/WekaExample/resources/datasets-arie_ben_david/ERA.arff";
	
	private static String data_set_2="/home/jcorvi/workspace/WekaExample/resources/weather.txt";
	
	private static String data_set_hep="/home/jcorvi/workspace/WekaExample/resources/datasets-UCI//UCI/hepatitis.arff";
	
	private static String data_set_diab="/home/jcorvi/workspace/WekaExample/resources/datasets-UCI//UCI/diabetes.arff";
	
	private static String data_set_zoo="/home/jcorvi/workspace/WekaExample/resources/datasets-UCI//UCI/zoo.arff";
	
	public static void main(String[] args) throws Exception {
		//BufferedReader datafile = WekaTest.readDataFile(data_set_diab);
		BufferedReader datafile = WekaTest.readDataFile(data_set_zoo);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		generateAssociations(data);
		
	}
	
	/**
	 * Consersion de campos Numericos a Nominales.
	 * @param dataProcessed
	 * @param variables
	 * @return
	 * @throws Exception
	 */
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
			System.out.println(data);
			Instances data3 = NumericToNominal(data,"2-18");
			System.out.println(data3);
			apriori.buildAssociations(data3);
			//apriori.buildAssociations(data);
			
			
			System.out.println(apriori);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	
}
