package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

public class OnlineLDA {
	
	private int K_;	// number of Cluster
	private double rhot_;	// rhot
	private double tau0_;	// 
	private double kappa_;	// 
	private double eta0_;	// 
	private double alpha_;
	private ArrayList<HashMap<String, double[]>> phi_;
	private ArrayList<double[]> gamma_;
	private Map<String, Double>[] lambda_;
	private Map<String, Double>[] lambdaBar_;
	
	// Random
	private final double gammaShape = 10;
	private final double gammaScale = 0.1;
	private final double initGetGamma = 0.1;
	GammaDistribution gamma = new GammaDistribution(gammaShape, gammaScale);
	
	
	// constructor
	public OnlineLDA(int k, double tau0, double kappa, double eta0, double alpha){
		K_ = k;
		tau0_ = tau0;
		kappa_ = kappa;
		eta0_  = eta0;
		alpha_ = alpha;
		initLambdas();
		initGamma();
		initPhi();
	}
	


	public void trainPerLine(Feature[] features, int time){
		int wordSize = features.length;
		this.rhot_ = Math.pow(tau0_ + time, -kappa_);
		
		initGammaPerLine(time);
		
		initPhiPerLine(features);
		
		checkAndInitializeLambdaPerLine(features);	
		
	
		HashMap<String, Integer> countMap = new HashMap<String, Integer>();

		for(int i=0, SIZE = features.length; i < SIZE; i++){
			countMap.put(features[i].getWord(), features[i].frequency_);
		}
		
		for(int tt=0; tt<100; tt++){
			//
			for(int w=0; w<wordSize; w++){
				for(int k=0; k<K_; k++){
					String tmpWord = features[w].getWord();
					phi_.get(time).get(tmpWord)[k] = calcEq(time, k, tmpWord);
				}
			}
			//
			for(int k=0; k<K_; k++){
				gamma_.get(time)[k] = calcGamma(time, k, features, countMap);				
			}
		}
		computeLambdaBar(time, features, countMap);
		
		updateLambda();
	}


	private void initPhi() {
		// Initialize phi_
		phi_ = new ArrayList<HashMap<String, double[]>>();

	}

	private void initGamma() {
		// Initialize gamma_
		gamma_ = new ArrayList<double[]>();

	}

	private void checkAndInitializeLambdaPerLine(Feature[] features) {
		ArrayList<String> newWords = new ArrayList<String>();
		int SIZE = features.length;
		for(int i=0; i<SIZE; i++){
			String tmpKey = features[i].getWord();
			if(!lambda_[0].containsKey(tmpKey)){
				newWords.add(tmpKey);
			}
		}
		

		int SIZExK_ = SIZE*K_;
		
		double[] tmp = getRandomGammaArray(SIZExK_);
		double[] tmp2 = getRandomGammaArray(SIZExK_);
		
		
		double value, value2;
		
		for(int i=0; i<SIZE; i++){
			String key = features[i].getWord();
			for(int j=0; j<K_; j++){
				int idx = i * SIZE + j;
				value = tmp[idx];
				value2 = tmp2[idx];
				lambda_[j].put(key, value);
				lambdaBar_[j].put(key, value2);
			}
		}
	}

	private double[] getRandomGammaArray(int sIZExK_) {
		double[] ret = new double[sIZExK_];
		for(int i=0; i<sIZExK_; i++){
			ret[i] = getRandomGamma();
		}
		return ret;
	}

	private double getRandomGamma() {
		double ret = 0;
		ret = Gamma.gamma(initGetGamma);
		return ret;
	}

	private void initPhiPerLine(Feature[] features) {
		int wordSize = features.length;
		HashMap<String, double[]> tmpMap = new HashMap<String, double[]>();

		for(int i=0; i<wordSize; i++){
			String str = features[i].getWord();
			double[] tmpDArray = getUniformalVector(wordSize, 1);
			tmpMap.put(str, tmpDArray);
		}
		phi_.add(tmpMap);
	}

	private void initGammaPerLine(int time) {
		double[] tmp = getUniformalVector(K_, 1);
		gamma_.add(tmp);
	}

	private double[] getUniformalVector(int wordSize, int initializer) {
		double[] ret = new double[wordSize];
		Arrays.fill(ret, initializer);
		return ret;
	}

	private void initLambdas() {
		lambda_ = new HashMap[K_];
		lambdaBar_ = new HashMap[K_];
		for(int k=0; k<K_; k++){
			lambda_[k] = new HashMap<String, Double>();
			lambdaBar_[k] = new HashMap<String, Double>();
		}
	}
	
	private void updateLambda() {
		
		for(int k=0; k<K_; k++){
			for(String key:lambda_[k].keySet()){
				double nextLambda_kw = (1 - rhot_) * lambda_[k].get(key) + this.rhot_ * lambdaBar_[k].get(key);
				lambda_[k].put(key, nextLambda_kw);
			}
		}
		
	}

	private void computeLambdaBar(int time, Feature[] features, HashMap<String, Integer> countMap){
		for(int k=0; k<K_; k++){
			for(String key:lambdaBar_[k].keySet()){
				double tmp= eta0_ + getD() * getN(countMap, key) * phi_.get(time).get(key)[k];		// TODO fix eta0 -> eta0
				lambdaBar_[k].put(key, tmp);
			}
		}
	}

	private int getD() {
		return 1;	// TODO temp
	}

	private double getN(HashMap<String, Integer> countMap, String key) {
		if(countMap.containsKey(key)){
			return countMap.get(key);
		}else{
			return 0;
		}
	}

	private double calcGamma(int time, int k, Feature[] features, HashMap<String, Integer> countMap) {
		double ret = alpha_;
		for(int w=0, W=features.length; w<W; w++){
			String tmpWord = features[w].getWord();
			ret += phi_.get(time).get(tmpWord)[k] * getN(countMap, tmpWord);
		}
		return ret;
	}

	private double calcEq(int time, int k, String word) {
		double ret = 0;
		
		double gammaSum = calcGammaSum(time);
		double tmpTheta = digamma(this.gamma_.get(time)[k]) - digamma(gammaSum);
		
		double lambdaSum = calcLambdaSum(k);
		double tmpBeta = digamma(this.lambda_[k].get(word)) - digamma(lambdaSum);

		ret = Math.exp(tmpTheta + tmpBeta);

		return ret;
	}

	private double digamma(double d) {
		double ret = 0;
		ret = Gamma.digamma(d);
		return ret;
	}

	private double calcLambdaSum(int k) {
		double ret = 0;
		for(String tmpStr:lambda_[k].keySet()){

			ret += lambda_[k].get(tmpStr);
		}
		return ret;
	}

	private double calcGammaSum(int time) {
		double ret = 0;
		for(int i=0; i<K_; i++){
			Object tmp = gamma_.get(time);	// tmp
			ret += gamma_.get(time)[i];
		}
		return ret;
	}


	public static final class Feature{
		private String word_;
		private int frequency_;
		public Feature(String w, int f){
			word_ = w;
			frequency_ = f;
		}
		
		public String getWord(){
			return word_;
		}
		public void setWord(String str){
			word_ = str;
		}
		
		public int getFrequency(){
			return frequency_;
		}
		public void setFrequency(int f){
			frequency_ = f;
		}
	}
}