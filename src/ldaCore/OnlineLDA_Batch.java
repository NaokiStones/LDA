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
	double shape = 10;
	double scale = 1;
	
	// Hyper Parameter 
	double alpha_= Double.NaN;	
	double eta_  = Double.NaN;
	double tau0_ = Double.NaN;
	double kappa_= Double.NaN;
	
	int ITERNUM_ = -1;	
	double CHANGE_THREASH_HOLD = 1E-5;
	
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
		lambda_ = new HashMap<String, double[]>(10000);
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
		
		clearPhi();
		initPhi(names, Nds, time);
		
		clearGamma();
		initGamma();
		
		
		for(int iter=0; iter < ITERNUM_; iter++){
			System.out.println("iter:" + iter);
			// E STEP
			// Gamma
			if(time >=1000){
				System.out.println("Gamma!");
			}
			for(int d=0; d<D_; d++){
				for(int k=0; k<K_; k++){
					String[] tmpNames = names[d];
					int tmpNd = Nds[d];
					gamma_.get(d)[k] = alpha_ + getSumForGamma(d, k, tmpNd, tmpNames);
				}
			}
			
			// Phi
//			if(time >= 1000){
//				System.out.println("iter:" + iter);
//			}
			for(int d=0; d<D_; d++){
//				if(time >=1000){
//					System.out.println("d:" + d);
//					System.out.println("Nds[d]:"+ Nds[d]);
//				}
				for(int w=0; w<Nds[d]; w++){
//					if(time >=1000){
//						System.out.println("w:" + w);
//					}
					String tmpWord = names[d][w];
//					long timeThetaS=0, timeThetaE=0, timeBetaS=0, timeBetaE=0;	// TODO remove
					double[] EqThetaVector = getDirichletVector(gamma_.get(d));
					for(int k=0; k<K_; k++){
						int tmpNd = Nds[d];
						String[] tmpNames = names[d]; 

//						if(time >=1000) timeThetaS = System.nanoTime();
//						double EqTheta = getEqTheta(d, k);		// TODO REMOVE OR 
//						if(time >=1000) timeThetaE = System.nanoTime();

//						if(time >=1000) timeBetaS= System.nanoTime();
						double EqBeta  =  getEqBeta(k, tmpWord, tmpNd, tmpNames);
//						if(time >=1000) timeBetaE= System.nanoTime();
						
//						if(time >= 1000){
//							System.out.println("theta:" + (timeThetaE - timeThetaS));
//							System.out.println("beta:" + (timeBetaE - timeBetaS));
//						}

						phi_.get(d).get(tmpWord)[k] = Math.exp(EqThetaVector[k] + EqBeta);
					}
				}
			}
		}
		
		
		
		if(time >= 1000){
			System.out.println("M Step!");
		}
		// M STEP
		for(int d=0; d<D_; d++){
			for(int w=0; w<Nds[d]; w++){
				String tmpWord = names[d][w];
				int tmpCount = featureBatch[d][w].getCount();
				for(int k=0; k<K_; k++){
					// Compute Lambda_Bar
					double tmpLambda_kw = eta_ + D_ * tmpCount * phi_.get(d).get(tmpWord)[k];
					// Set Lambda
					lambda_.get(tmpWord)[k] = (1 - rhot) * lambda_.get(tmpWord)[k] + rhot * tmpLambda_kw;
					//						System.out.println(lambda_.get(tmpWord)[k]);
					//						lambda_.get(tmpWord)[k] = 0;
				}
			}
		}
	}
	

	private double[] getDirichletVector(double[] ds) {
		double[] ret = new double[K_]; 
		double sum = 0;
		for(int k=0; k<K_; k++){
			sum += ds[k];
		}
		double digammaSum = Gamma.digamma(sum);
		for(int k=0; k<K_; k++){
			ret[k] = Gamma.digamma(ds[k]) - digammaSum;
		}
		return ret;
	}

	private void clearPhi() {
		phi_ = new ArrayList<HashMap<String, double[]>>();
	}

	private void clearGamma() {
		gamma_ = new ArrayList<double[]>();
	}

	private double getSumForGamma(int d, int k, int Nd, String[] tmpNames) {
		double ret = 0;
		for(int i=0, W=Nd; i<W; i++){
			String tmpName = tmpNames[i];
			ret += phi_.get(d).get(tmpName)[k];
		}
		return ret;
	}


	private double getEqBeta(int k, String tmpWord, int tmpNd, String[] tmpNames) {
		double ret = 0;
//		long timeSumLambdaS=0, timeSumLambdaE=0, timeDigammaS=0,timeDigammaE=0;
//		timeSumLambdaS = System.nanoTime();
		double sumLambda = getSumLambda(k, tmpNd, tmpNames);
//		timeSumLambdaE = System.nanoTime();
		
		double tmpLambda_wk = lambda_.get(tmpWord)[k];

//		timeDigammaS= System.nanoTime();
		ret = Gamma.digamma(tmpLambda_wk) - Gamma.digamma(sumLambda);
//		timeDigammaE= System.nanoTime();

//		System.out.println("Digamma:" + (timeDigammaE - timeDigammaS));
//		System.out.println("Lambda:" + (timeSumLambdaE- timeSumLambdaS));
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


	private double getEqTheta(int d, int k) {
		double ret = 0;
		double sumGamma = getSumGamma(d);
		ret += getDigamma(gamma_.get(d)[k]) - getDigamma(sumGamma); 
		return ret;
	}


	private double getDigamma(double d) {
		double ret = 0;
		ret += Gamma.digamma(d);
		return ret;
	}


	private double getSumGamma(int d) {
		double ret = 0;
		for(int k=0; k<K_; k++){
			ret += gamma_.get(d)[k];
		}
		return ret;
	}


	private void initGamma() {
		for(int i=0; i<D_; i++){
			double[] tmpDArray = getUniformalDArray(1);
			gamma_.add(tmpDArray);
		}
	}


	private void initPhi(String[][] names, int[] Nds, int time) {
		for(int d=0; d<D_; d++){
			HashMap<String, double[]> tmpPhi = new HashMap<String, double[]>(100);
			phi_.add(tmpPhi);

			for(int w=0; w<Nds[d]; w++){
				String tmpName = names[d][w];
				double[] tmpDArray = getUniformalDArray(0);
				phi_.get(d).put(tmpName, tmpDArray);
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
		System.out.println("show Topic Words");
		System.out.println("lambda_.size()" + lambda_.size());

		for(int k=0; k<K_; k++){
			System.out.print("Topic:" + k);
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("===================================");
			ArrayList<String> sortedWords = getSortedLambda(k);
			System.out.println("k:" + k + " sortedWords.size():" + sortedWords.size());
			for(int tt=0; tt<50; tt++){
				String tmpWord = sortedWords.get(tt);
				System.out.println("No." + tt + "\t" +tmpWord + ":\t" + lambda_.get(tmpWord)[k]);
			}
			System.out.println("==========================================");
			
		}
	}

	private ArrayList<String> getSortedLambda(int k) {
		ArrayList<String> ret = new ArrayList<String>();
		ArrayList<LambdaCompare> compareList = new ArrayList<LambdaCompare>();
		for(String word:lambda_.keySet()){
			double tmpValue = lambda_.get(word)[k];
//			System.out.println("word:" + word);
//			System.out.println("tmpValue:" + tmpValue);

			compareList.add(new LambdaCompare(tmpValue, word));
		}

		Collections.sort(compareList, new LambdaComparator());
		
		for(int w=0,W=compareList.size(); w<W; w++){
			ret.add(compareList.get(w).getName());
		}
		return ret;
	}
	
}