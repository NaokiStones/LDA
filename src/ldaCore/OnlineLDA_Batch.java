package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Collections;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import com.sun.xml.internal.ws.addressing.W3CAddressingConstants;

import utils.LambdaComparator;
import utils.LambdaCompare;

public class OnlineLDA_Batch implements LDAModel{
	static int totalD = 0;
	static int D_ = -1;	
	static int K_ = -1;	
	static int W_ = -1;
	private HashMap<String, double[]> lambda_;	
	private ArrayList<HashMap<String, double[]>> phi_;	
	private ArrayList<double[]> gamma_;
	
	
	// TEMP CONSTANTS
	double shape = 100;	// 100
	double scale = 0.01;	// 0.01
	
	// Hyper Parameter 
	double alpha_= Double.NaN;	
	double eta_  = Double.NaN;
	double tau0_ = Double.NaN;
	double kappa_= Double.NaN;
	
	int ITERNUM_ = -1;	
	double CHANGE_THREASH_HOLD = 1E-5;
	
	double perplexity = 0;
	
	static int batchVSize = 0;
	
	// RANDOM
	GammaDistribution gd = new GammaDistribution(shape, scale);
	Random rnd = new Random(11111111);
	
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
//		double rhot = 0.01;2

		D_ = featureBatch.length;
		totalD += D_;
		
		int[] Nds; 
		String[][] names;
		Nds = getNds(featureBatch);	// get Number of word in each document 
		names = getNames(featureBatch, Nds);	// get names
		
		setBatchVocabularySize(Nds);
		
		initLambda(names, Nds);
		
		clearPhi();
		initPhi(names, Nds, time);
		
		clearGamma();
		initGamma();
		
		ArrayList<double[]> lastGamma = copyGamma();
		double[][] EqThetaVector = new double[D_][];
		HashMap<String, double[]> betas = new HashMap<String, double[]>();
		
		for(int iter=0; iter < ITERNUM_; iter++){
//			System.out.println("iter:" + iter);
			lastGamma = new ArrayList<double[]>();
			lastGamma = copyGamma();
			
			
			// E STEP
			// Gamma
			for(int d=0; d<D_; d++){
//				double tmpGammaG = 0;
				for(int k=0; k<K_; k++){
					String[] tmpNames = names[d];
					int tmpNd = Nds[d];
					double tmpGammaGK = alpha_ + getSumForGamma(d, k, tmpNd, tmpNames);
					gamma_.get(d)[k] = tmpGammaGK;
//					tmpGammaG += tmpGammaGK;
				}
				
//				for(int k=0; k<K_; k++){
//					gamma_.get(d)[k] /= tmpGammaG;
//				}
			}
			
			// Phi
			for(int d=0; d<D_; d++){
				EqThetaVector[d] = getEqThetaVector(gamma_.get(d));
//				double tmpPhiSum[] = new double[Nds[d]];
//				Arrays.fill(tmpPhiSum, 0d);
				double tmpPhi = 0;
				for(int k=0; k<K_; k++){
					String[] tmpNames = names[d]; 
					int tmpNd = Nds[d];
					double tmpSumLambda = getSumLambda(k, tmpNd, tmpNames);
					double EqThetaVectorK = EqThetaVector[d][k];
					for(int w=0; w<Nds[d]; w++){
						String tmpWord = names[d][w];
						double EqBeta  =  getEqBeta(k, tmpWord, tmpNd, tmpSumLambda); //						if(time >=1000) timeBetaE= System.nanoTime(); 
						if(betas.containsKey(tmpWord)){
							double[] tmp = betas.get(tmpWord);
							tmp[k] = EqBeta;
							betas.put(tmpWord, tmp);
						}else{
							double[] tmp = getUniformalDArray(0);
							tmp[k] = EqBeta;
							betas.put(tmpWord, tmp);
						}
						tmpPhi = Math.exp(EqThetaVectorK + EqBeta);
						phi_.get(d).get(tmpWord)[k] = tmpPhi + 1E-100;
//						System.out.print("next phi[d][w]:" + tmpPhi);
//						System.out.print(tmpPhi + ",");
//						tmpPhiSum[w] += tmpPhi;
					}
				}
				
//				for(int k=0; k<K_; k++){
//					for(int w=0; w<Nds[d]; w++){
//						String tmpWord = names[d][w];
//						phi_.get(d).get(tmpWord)[k] /= tmpPhiSum[w];
//					}
//				}
			}

			if(notChangeGamma(lastGamma, gamma_)){
//				System.out.println("converged:" + iter);
				break;
			}
		}


		
		// M STEP
		for(int d=0; d<D_; d++){
			for(int w=0; w<Nds[d]; w++){
				String tmpWord = names[d][w];
				int tmpCount = featureBatch[d][w].getCount();
//				double tmpLambdaWord = 0;
				for(int k=0; k<K_; k++){
					// Compute Lambda_Bar
					double DNTheta = D_ * tmpCount * phi_.get(d).get(tmpWord)[k];
					double tmpLambda_kw = eta_ + DNTheta;
//					System.out.println("D_:" + D_);
//					System.out.println("tmpCount:" + tmpCount);
//					System.out.println("phi_.get(d).get(tmpWord)[k]:" + phi_.get(d).get(tmpWord)[k]);
//					System.out.println("DNTheta:" + DNTheta);
//					// Set Lambda
					double nextLambda = (1 - rhot) * lambda_.get(tmpWord)[k] + rhot * tmpLambda_kw;
					lambda_.get(tmpWord)[k] = nextLambda;
//					tmpLambdaWord += nextLambda;
//						System.out.println(lambda_.get(tmpWord)[k]);
//						lambda_.get(tmpWord)[k] = 0;
				}
//				for(int k=0; k<K_; k++){
//					lambda_.get(tmpWord)[k] /= tmpLambdaWord;
//				}
			}
		}
		
		// calc Perplexity for mini-batch and Train
		
		perplexity += Math.exp(calcBoundPerBatch(Nds, featureBatch, EqThetaVector, betas) / (-1.0*batchVSize));
	}
	
	private void setBatchVocabularySize(int[] nds) {
		batchVSize = 0;
		for(int i=0, SIZE=nds.length; i<SIZE; i++){
			batchVSize += nds[i];
		}
//		System.out.println("batchVSize:" + batchVSize);
	}

	private double calcBoundPerBatch(int[] Nds, Feature[][] featureBatch, double[][] EqThetaVector,HashMap<String, double[]> EqBetas){
		double ret = 0;
		
		String tmpWord;
		
		double tmp = 0;
		
		double tmpSum  = 0;
		double tmpSum2 = 0;
		double tmpSum3 = 0;
		double tmpSum4 = 0;

		double tmpSum3_1 = 0;
		double tmpSum3_2 = 0;
		
		double tmpSum4_1 = 0;
		double tmpSum4_2 = 0;
		double tmpSum4_3 = 0;
		double tmpSum4_3_1 = 0;
		double tmpSum4_3_2 = 0;
		
		for(int d=0; d<D_; d++){
			for(int w=0; w<Nds[d]; w++){
				tmpWord = featureBatch[d][w].getName();
				for(int k=0; k<K_; k++){
					double tmp_Phi_dk = phi_.get(d).get(tmpWord)[k];
					ret += tmp_Phi_dk * (EqThetaVector[d][k] - EqBetas.get(tmpWord)[k] - Math.log(tmp_Phi_dk));
				}
			}
			// END OF FIRST LINE

			tmpSum = 0;
			tmpSum2 = 0;
			for(int k=0; k<K_; k++){
				tmpSum += gamma_.get(d)[k];
				
				tmpSum2 += ((alpha_ - gamma_.get(d)[k]) * EqThetaVector[d][k] + Math.log(Gamma.gamma(gamma_.get(d)[k])));
				
			}
			ret -= Math.log(tmpSum);
			ret += tmpSum2;

			// END OF SECOND LINE
			tmpSum3 = 0;
			
			String[][] names;
			names = getNames(featureBatch, Nds);	// get names
			String[] tmpNames = names[d]; 
			

			for(int k=0; k<K_; k++){
				tmpSum3_1 = 0;
				tmp = 0;
				for(int w=0; w<Nds[d]; w++){
					tmpWord = featureBatch[d][w].getName();
					tmp += lambda_.get(tmpWord)[k];
//					System.out.println("Lambda_kw:" + lambda_.get(tmpWord)[k]);
				}
				tmpSum3_1 = (-1) * Math.log(Gamma.gamma(tmp));
				tmpSum3 += tmpSum3_1;
				
				int tmpNd = Nds[d];
				double tmpSumLambda = getSumLambda(k, tmpNd, tmpNames);
				tmpSum3_2 = 0;
				for(int w=0; w<Nds[d]; w++){
					tmpWord = featureBatch[d][w].getName();
					tmpSum3_2 += (eta_ - lambda_.get(tmpWord)[k] ) * getEqBeta(k, tmpWord, tmpNd, tmpSumLambda) + Math.log(Gamma.gamma(lambda_.get(tmpWord)[k]));
				}
				tmpSum3 += tmpSum3_2;
			}
			
			tmpSum3 /= D_;
			ret += tmpSum3;
			// END OF THIRD LINE
			
			tmpSum4_1 = Math.log(K_ * alpha_);
			tmpSum4_2 = K_ * Math.log(Gamma.gamma(alpha_));
			tmpSum4_3_1 = Math.log(Gamma.gamma(W_ * eta_));
			tmpSum4_3_2 = W_ * Math.log(Gamma.gamma(eta_));
			
			tmpSum4_3 = (tmpSum4_3_1 - tmpSum4_3_2) / D_;
			tmpSum4 = tmpSum4_1 - tmpSum4_2 + tmpSum4_3;
			ret += tmpSum4;
		}

		return ret;
	}


//	private double calcPerplexityPerBatch(int[] Nds, Feature[][] featureBatch) {
//		double ret = 0;
//		double tmpSum = 0;
//		double Pwd = 1;
//		int ndPrint = 0;	// TODO remove
//		for(int d=0; d<D_; d++){
//			Pwd = 1;
//			for(int n=0,Nd = Nds[d]; n<Nd; n++){
//				ndPrint = Nd;
//				tmpSum = 0;
//				String word = featureBatch[d][n].getName();
//				for(int k=0; k<K_; k++){
//					tmpSum += gamma_.get(d)[k] * phi_.get(d).get(word)[k];
//				}
//				Pwd += Math.log(tmpSum);
//			}
//			ret += (Pwd);
//
//		}
//		
//		double NdSum = 0;
//		for(int d=0; d<D_; d++){
//			NdSum += Nds[d];
//		}
//		
//		
//		ret = Math.exp(((-1) * (ret))/ (NdSum));
////		System.out.println("ret:" + ret);
//		ndPrint +=1;
//
//		return ret;
//	}

	private ArrayList<double[]> copyGamma() {
		ArrayList<double[]> ret = new ArrayList<double[]>();
		
		for(int d=0; d<D_; d++){
			double[] tmp = new double[K_];
			for(int k=0; k<K_; k++){
				tmp[k] = gamma_.get(d)[k];
			}
			ret.add(tmp);
		}
		return ret;
	}

	private boolean notChangeGamma(ArrayList<double[]> lastGamma, ArrayList<double[]> gamma_time) {
		double diff = 0;
		
		for(int d=0; d<D_; d++){
			for(int k=0; k<K_; k++){
				diff += Math.abs(lastGamma.get(d)[k] - gamma_time.get(d)[k]);
			}
		}

		
		if(diff < CHANGE_THREASH_HOLD * K_ * D_){
			return true;
		}else{
			return false;
		}
	}

	private double[] getEqThetaVector(double[] ds) {
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


	private double getEqBeta(int k, String tmpWord, int tmpNd, double sumLambda) {
		double ret = 0;
//		long timeSumLambdaS=0, timeSumLambdaE=0, timeDigammaS=0,timeDigammaE=0;
//		timeSumLambdaS = System.nanoTime();
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
//			double[] tmpDArray = getUniformalDArray(1);
			double[] tmpDArray = getUniformalRandomDArray();
			gamma_.add(tmpDArray);
		}
	}
	


	private double[] getUniformalRandomDArray() {
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = rnd.nextGaussian() + 1.0;
		}
		
		return ret;
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
					W_ = lambda_.size();
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
			
			double lambdaSum = 0;
			for(String key:lambda_.keySet()){
				lambdaSum += lambda_.get(key)[k];
			}
				
			
			System.out.print("Topic:" + k);
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("===================================");
			ArrayList<String> sortedWords = getSortedLambda(k);
			System.out.println("k:" + k + " sortedWords.size():" + sortedWords.size());
			for(int tt=0; tt<50; tt++){
				String tmpWord = sortedWords.get(tt);
				System.out.println("No." + tt + "\t" +tmpWord + ":\t" + lambda_.get(tmpWord)[k] / lambdaSum);
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

	public double getPerplexity() {
		double ret = perplexity;
		perplexity = 0;
		return ret;
	}
}