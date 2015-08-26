package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Collections;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;


import utils.LambdaComparator;
import utils.LambdaCompare;

public class OnlineLDA_Batch implements LDAModel{
	static int D_ = -1;	
	static int K_ = -1;	
	private HashMap<String, double[]> lambda_;	
	private ArrayList<HashMap<String, double[]>> phi_;	
	private ArrayList<double[]> gamma_;
	
	
	// TEMP CONSTANTS
	double shape = 100;
	double scale = 1;
	
	// Hyper Parameter 
	double alpha_= Double.NaN;	
	double eta_  = Double.NaN;
	double tau0_ = Double.NaN;
	double kappa_= Double.NaN;
	
	int ITERNUM_ = -1;	
	
	// RANDOM
	GammaDistribution gd = new GammaDistribution(shape, scale);
	
	public OnlineLDA_Batch(int K, double alpha, double eta, double tau0, double kappa, int Iternum){
		K_ = K;
		alpha_ = alpha;
		eta_   = eta;
		tau0_  = tau0;
		kappa_ = kappa;
		ITERNUM_ = Iternum;
		
		
		// FIRST INITIALIZE
		lambda_ = new HashMap<String, double[]>();
		phi_    = new ArrayList<HashMap<String, double[]>>();
		gamma_  = new ArrayList<double[]>();
		
	}
	
	@Override
	public void trainBatch(Feature[][] featureBatch, int time){	/* ************************************ ************************* ******************* ***************** */
		double rhot = Math.pow(tau0_ + time, -kappa_);
		D_ = featureBatch.length;
		
		int[] Nds; 
		String[][] names;
		Nds = getNds(featureBatch);	// get Number of word in each document 
		names = getNames(featureBatch, Nds);	// get names
		
		initLambda(names, Nds);
		initPhi(names, Nds, time);
		
		
		for(int iter=0; iter < ITERNUM_; iter++){
			// E STEP
			// Phi
			if(time >= 1000){
				System.out.println("iter:" + iter);
			}
			for(int d=0; d<D_; d++){
				if(time >=1000){
					System.out.println("d:" + d);
					System.out.println("Nds[d]:"+ Nds[d]);
				}
				initGamma();
				int tmpTime = time + d;
				for(int w=0; w<Nds[d]; w++){
					if(time >=1000){
						System.out.println("w:" + w);
					}
					String tmpWord = names[d][w];
					long timeThetaS=0, timeThetaE=0, timeBetaS=0, timeBetaE=0;	// TODO remove
					for(int k=0; k<K_; k++){
						int tmpNd = Nds[d];
						String[] tmpNames = names[d]; 

//						if(time >=1000) timeThetaS = System.nanoTime();
						double EqTheta = getEqTheta(tmpTime, k);
//						if(time >=1000) timeThetaE = System.nanoTime();

//						if(time >=1000) timeBetaS= System.nanoTime();
						double EqBeta  =  getEqBeta(k, tmpWord, tmpNd, tmpNames);
//						if(time >=1000) timeBetaE= System.nanoTime();
						
//						if(time >= 1000){
//							System.out.println("theta:" + (timeThetaE - timeThetaS));
//							System.out.println("beta:" + (timeBetaE - timeBetaS));
//						}
						
						phi_.get(tmpTime).get(tmpWord)[k] = Math.exp(EqTheta + EqBeta);
					}
				}
			}
			// Gamma
			if(time >=1000){
				System.out.println("Gamma!");
			}
			for(int d=0; d<D_; d++){
				int tmpTime = time + d;
				for(int k=0; k<K_; k++){
					String[] tmpNames = names[d];
					int tmpNd = Nds[d];
					gamma_.get(tmpTime)[k] = alpha_ + getSumForGamma(tmpTime, k, tmpNd, tmpNames);
				}
			}
			
			if(time >= 1000){
				System.out.println("M Step!");
			}
			// M STEP
			double divider = 1d / D_;
			for(int d=0; d<D_; d++){
				int tmpTime = time + d;
				for(int w=0; w<Nds[d]; w++){
					String tmpWord = names[d][w];
					int tmpCount = featureBatch[d][w].getCount();
					for(int k=0; k<K_; k++){
						// Compute Lambda_Bar
						double tmpLambda_kw = eta_ + divider * tmpCount * phi_.get(tmpTime).get(tmpWord)[k];
						// Set Lambda
						lambda_.get(tmpWord)[k] = (1 - rhot) * lambda_.get(tmpWord)[k] + rhot * tmpLambda_kw;
					}
				}
			}
		}
	}
	

	private double getSumForGamma(int tmpTime, int k, int Nd, String[] tmpNames) {
		double ret = 0;
		for(int i=0, W=Nd; i<W; i++){
			String tmpName = tmpNames[i];
			ret += phi_.get(tmpTime).get(tmpName)[k];
		}
		return ret;
	}


	private double getEqBeta(int k, String tmpWord, int tmpNd, String[] tmpNames) {
		double ret = 0;
		double sumLambda = getSumLambda(k, tmpNd, tmpNames);
		ret = getDigamma(lambda_.get(tmpWord)[k]) + getDigamma(sumLambda);
		return ret;
	}


	private double getSumLambda(int k, int tmpNd, String[] names) {
		double ret = 0;
		for(int w=0, W=tmpNd; w<W; w++){
			String tmpName = names[w];
			ret += lambda_.get(tmpName)[k];
		}
		return ret;
	}


	private double getEqTheta(int tmpTime, int k) {
		double ret = 0;
		double sumGamma = getSumGamma(tmpTime);
		ret += getDigamma(gamma_.get(tmpTime)[k]) + getDigamma(sumGamma); 
		return ret;
	}


	private double getDigamma(double d) {
		double ret = 0;
		ret += Gamma.digamma(d);
		return ret;
	}


	private double getSumGamma(int tmpTime) {
		double ret = 0;
		for(int k=0; k<K_; k++){
			ret += gamma_.get(tmpTime)[k];
		}
		return ret;
	}


	private void initGamma() {
		double[] tmpDArray = getUniformalDArray(1);
		gamma_.add(tmpDArray);
	}


	private void initPhi(String[][] names, int[] Nds, int time) {
		for(int d=0; d<D_; d++){
			int tmpTime = time + d;
			HashMap<String, double[]> tmpPhi = new HashMap<String, double[]>();
			phi_.add(tmpPhi);

			for(int w=0; w<Nds[d]; w++){
				String tmpName = names[d][w];
				double[] tmpDArray = getUniformalDArray(0);
				phi_.get(tmpTime).put(tmpName, tmpDArray);
			}
		}
	}


	private double[] getUniformalDArray(double initializer) {
		double[] ret = new double[K_];
		Arrays.fill(ret, initializer);
		return ret;
	}


	private void initLambda(String[][] names, int[] Nds) {
		for(int i=0; i<D_; i++){
			for(int j=0; j<Nds[i]; j++){
				String tmpKey = names[i][j];
				if(!lambda_.containsKey(tmpKey)){
					double[] tmp = getRandomGammaArray();
					lambda_.put(tmpKey, tmp);
				}
			}
		}
	}


	private double[] getRandomGammaArray() {
		double[] ret = new double[K_];
		double tmpSum = 0;
		for(int i=0; i<K_; i++){
			ret[i] = getGamma();
			tmpSum += ret[i];
		}
		
		// Sum Up to 1
		for(int i=0; i<K_; i++){
			ret[i] /= tmpSum;
		}
		
		return ret;
	}


	private double getGamma() {
		// TODO FIX later
		double ret = gd.sample();
		return ret;
	}


	private String[][] getNames(Feature[][] featureBatch, int[] Nds) {
		String[][] ret = new String[D_][];
		
		for(int i=0; i<D_; i++){
			int Nd = Nds[i];
			ret[i] = new String[Nd];
//			System.out.println("Nd:" + Nd);
			for(int j=0; j<Nd; j++){
//				System.out.println("j:" + j);
				ret[i][j] = featureBatch[i][j].getName();
			}
		}
				
		return ret;
	}


	private int[] getNds(Feature[][] featureBatch) {
		int[] ret = new int[D_];
		for(int i=0; i<D_; i++){
			ret[i] = featureBatch[i].length;
		}
		return ret;
	}


	public static class Feature{	// TODO OK?
		private String _name= "Not initialized";
		private int   _count= -1;
		
		public Feature(String name, int count){
			_name = name;
			_count= count;
		}
		
		public String getName(){
			return _name;
		}
		
		public int getCount(){
			return _count;
		}
	}


	@Override
	public void showTopicWords() {
		for(int k=0; k<K_; k++){
			System.out.println("Topic:" + k);
			ArrayList<String> sortedWords = getSortedLambda(k);
			System.out.println("k:" + k + " sortedWords.size():" + sortedWords.size());
			for(int tt=0; tt<20; tt++){
				String tmpWord = sortedWords.get(tt);
				System.out.println(tmpWord + ":" + lambda_.get(tmpWord)[k]);
			}
		}
	}

	private ArrayList<String> getSortedLambda(int k) {
		ArrayList<String> ret = new ArrayList<String>();
		ArrayList<LambdaCompare> compareList = new ArrayList<LambdaCompare>();
		for(String word:lambda_.keySet()){
			System.out.println(word);
			double tmpValue = lambda_.get(word)[k];
			compareList.add(new LambdaCompare(tmpValue, word));
		}
		Collections.sort(compareList, new LambdaComparator());
		return ret;
	}
	
}