package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;

import java.util.Collections;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;
import org.omg.CORBA.SystemException;

import utils.LambdaComparator;
import utils.LambdaCompare;

public class OnlineLDA_Batch implements LDAModel{
	
	// 
	private static double SHAPE = 5;
	private static double SCALE = 1 / SHAPE;
	private static int ITERNUM= 200;	// TODO INITIALIZE 
	private static double THRESHHOLD = 1E-7; // TODO INITIALIZE
	
	private static int D_;
	private static double estimateD = 80; // TODO INITIALIZE	
	private static int K_;	
	private static double alpha_;	
	private static double tau0_;	
	private static double kappa_;	
	private static double eta_;		
	
	private static HashMap<String, double[]> lambda_;
	private ArrayList<HashMap<String, double[]>> phi_;
	private ArrayList<double[]> gamma_;
	private ArrayList<double[]> eLogTheta;
	private ArrayList<double[]> expELogTheta;
	private HashMap<String, double[]> eLogBeta;	// TODO INITIALIZE
	private HashMap<String, double[]> expELogBeta;// TODO INITIALIZE
	
	
	// Perplexity
	private double perplexity = 0;
	
	

	// Random
	private GammaDistribution gd = new GammaDistribution(SHAPE, SCALE);
	private Random rnd = new Random(10001);
	
	
	public OnlineLDA_Batch(int K, double alpha, double eta, double tau0, double kappa, int IterNum) {
		K_ = K;
		alpha_ = alpha;
		eta_ = eta;
		tau0_ = tau0;
		kappa_= kappa;
		ITERNUM = IterNum;
		
		// lambda
		lambda_ = new HashMap<String, double[]>();
		// elogBeta expElogBeta
		eLogBeta = new HashMap<String, double[]>();
		expELogBeta = new HashMap<String, double[]>();

	}
	
	private double calcBoundPerBatch(Feature[][] featureBatch){
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

		double logGamma_dk = 0; 
		
		for(int d=0; d<D_; d++){
			for(int w=0; w<featureBatch[d].length; w++){
				tmpWord = featureBatch[d][w].getName();
				int ndw = featureBatch[d][w].getCount();
				for(int k=0; k<K_; k++){
					double tmp_Phi_dk = phi_.get(d).get(tmpWord)[k];
					ret += ndw * tmp_Phi_dk * (eLogTheta.get(d)[k] + eLogBeta.get(tmpWord)[k] - Math.log(tmp_Phi_dk));
				}
			}
			// END OF FIRST LINE

			tmpSum = 0;
			tmpSum2 = 0;
			for(int k=0; k<K_; k++){
				tmpSum += gamma_.get(d)[k];
				logGamma_dk = Math.log(Gamma.gamma(gamma_.get(d)[k]));

				tmpSum2 += ((alpha_ - gamma_.get(d)[k]) * eLogTheta.get(d)[k] + logGamma_dk);
			}
			ret -= Math.log(Gamma.gamma(tmpSum));
			ret += tmpSum2;

			// END OF SECOND LINE
			tmpSum3 = 0;
			
//			System.out.println("Print Lambda");
			for(int k=0; k<K_; k++){
				tmpSum3_1 = 0;

				System.out.println("w Size" + featureBatch.length);
				System.out.println("featureBatch:" + featureBatch[0].length);
				tmp = 0;
				for(int w=0; w<featureBatch[d].length; w++){
					tmpWord = featureBatch[d][w].getName();
					tmp += lambda_.get(tmpWord)[k];
					System.out.println(lambda_.get(tmpWord)[k]);
//					System.out.println("");
//					System.out.print("[" + k + ". " + tmpWord + "]:" + lambda_.get(tmpWord)[k]);
				}
//				System.out.println("");
				tmpSum3_1 = (-1) * Math.log(Gamma.gamma(tmp));
				tmpSum3 += tmpSum3_1;
				
				int tmpNd = featureBatch[d].length;
				double tmpSumLambda = 0;
				for(String key:lambda_.keySet()){
					tmpSumLambda += lambda_.get(key)[k];
				}
				tmpSum3_2 = 0;
				for(int w=0; w<featureBatch[d].length; w++){
					tmpWord = featureBatch[d][w].getName();
					tmpSum3_2 += (eta_ - lambda_.get(tmpWord)[k]) * eLogBeta.get(tmpWord)[k] + Math.log(Gamma.gamma(lambda_.get(tmpWord)[k]));
				}
				tmpSum3 += tmpSum3_2;
			}
			
			tmpSum3 /= D_;
			ret += tmpSum3;
			// END OF THIRD LINE
			
			tmpSum4_1 = Math.log(Gamma.gamma(K_ * alpha_));
			tmpSum4_2 = K_ * Math.log(Gamma.gamma(alpha_));
			tmpSum4_3_1 = Math.log(Gamma.gamma(lambda_.size() * eta_));
			tmpSum4_3_2 = lambda_.size() * Math.log(Gamma.gamma(eta_));
			
			tmpSum4_3 = (tmpSum4_3_1 - tmpSum4_3_2) / D_;
			tmpSum4 = tmpSum4_1 - tmpSum4_2 + tmpSum4_3;
			ret += tmpSum4;
		}

		return ret;
	}


	@Override
	public void trainBatch(Feature[][] featureBatch, int time) {
		D_ = featureBatch.length;
		double rhot = Math.pow(tau0_ + time, -kappa_);
		long start = System.nanoTime();
		checkNewFeature(featureBatch);
		long end = System.nanoTime();
		System.out.println("CHECK TIME(ALL):" + (end - start));
		
//		phi_ = new ArrayList<HashMap<String, double[]>>();

		start = System.nanoTime();
		for(int d=0; d < D_; d++){
			double[] gammaD = gamma_.get(d); 
			double[] elogThetaD = eLogTheta.get(d);
			double[] expElogThetaD = expELogTheta.get(d);
			
			HashMap<String, double[]> tmpPhi = phi_.get(d);

			for(int w=0,Nd=featureBatch[d].length; w<Nd; w++){
				String tmpName = featureBatch[d][w].getName();
				double[] tmp = new double[K_];
				for(int k=0; k<K_; k++){
					tmp[k] = expElogThetaD[k] * expELogBeta.get(tmpName)[k] + 1E-100;
				}
				tmpPhi.put(tmpName, tmp);
			}
			
			double[] lastGamma;
			// E Step
			for(int it=0; it<ITERNUM; it++){
				lastGamma = copyGamma(gammaD);
				
				// Gamma
				double tmpSumGammaD = 0;
				for(int k=0; k<K_; k++){
					double sumPhi_twk_N_tw = 0;
					for(int w=0, Nd = featureBatch[d].length; w<Nd; w++){
						String tmpKey = featureBatch[d][w].getName();
						int cnt = featureBatch[d][w].getCount();
						sumPhi_twk_N_tw += (tmpPhi.get(tmpKey)[k] * cnt);
					}
					gammaD[k] = alpha_ +(sumPhi_twk_N_tw); 
//					System.out.print(" [" + k + "]:" + gammaD[k]);
					tmpSumGammaD += gammaD[k];
				}
//				System.out.println("");
				for(int k=0; k<K_; k++){
					gammaD[k] /= tmpSumGammaD;
				}
				
				// Phi
				elogThetaD = getEqlogThetaArray(gammaD);
				for(int k=0; k<K_; k++) expElogThetaD[k] = Math.exp(elogThetaD[k]);
				for(int w=0, Nd = featureBatch[d].length; w<Nd; w++){
					String tmpName = featureBatch[d][w].getName();
					for(int k=0; k<K_; k++){
						tmpPhi.get(tmpName)[k] = expElogThetaD[k] * expELogBeta.get(tmpName)[k] + 1E-100;
					}
				}
				
				if(changeGamma(lastGamma, gammaD)){
					System.out.println("ITERATION:" + it);
					break;
				}
				// CALC Eqs 
//				start = System.nanoTime();
				updateEqThetas();
//				end = System.nanoTime();
//				System.out.println("eqTheta:" + (end -start));
//				start = System.nanoTime();
				updateEqEta();
//				end = System.nanoTime();
//				System.out.println("eqEta:" + (end -start));
			}
			
//			for(String key:tmpPhi.keySet()){
//				System.out.println(Arrays.toString(tmpPhi.get(key)));
//			}
			
			phi_.set(d, tmpPhi);
			eLogTheta.set(d, elogThetaD);
			expELogTheta.set(d, expElogThetaD);

			end = System.nanoTime();
			System.out.println("AFTER CHECK(ALL):" + (end - start));
		}
		
		// M Step
		HashMap<String, double[]> lambdaBar = new HashMap<String, double[]>();
		for(int d=0; d<D_; d++){
			for(int w=0, Nd=featureBatch[d].length; w<Nd; w++){
				String tmpName = featureBatch[d][w].getName();
				int tmpCount   = featureBatch[d][w].getCount();
				if(lambdaBar.containsKey(tmpName)){
					double[] tmp = lambdaBar.get(tmpName);
					for(int k=0; k<K_; k++){
						tmp[k] += eta_ + (D_) * tmpCount * phi_.get(d).get(tmpName)[k];
					}
					lambdaBar.put(tmpName, tmp);
				}else{
					double[] tmp = lambda_.get(tmpName);
					Arrays.fill(tmp, eta_);
					lambdaBar.put(tmpName, tmp);
				}
			}
		}
		
		HashSet<String> tmpSet = new HashSet<String>();
		for(String key: lambda_.keySet()){
			tmpSet.add(key);
			if(lambdaBar.containsKey(key)){
				double[] tmpLambdaArray = lambda_.get(key);
				double[] tmpLambdaBarArray = lambdaBar.get(key);
				for(int k=0; k<K_; k++){
					tmpLambdaArray[k] = (1 - rhot) * tmpLambdaArray[k] + rhot * tmpLambdaBarArray[k];
				}
				lambda_.put(key, tmpLambdaArray);
			}else{
				double[] tmpLambdaArray = lambda_.get(key);
				for(int k=0; k<K_; k++){
					tmpLambdaArray[k] = (1 - rhot) * tmpLambdaArray[k];				
				}			
				lambda_.put(key, tmpLambdaArray);
			}
		}
		for(String key:lambdaBar.keySet()){
			if(tmpSet.contains(key)){
				continue;
			}
			double[] tmp = lambdaBar.get(key);
			for(int k=0; k<K_; k++) tmp[k] = rhot*tmp[k];

			lambda_.put(key, tmp);
		}
		
		perplexity += calcBoundPerBatch(featureBatch);
	}


	private void updateEqEta() {
		double[] LambdaSums = new double[K_];
		Arrays.fill(LambdaSums, 0);
		for(String key:eLogBeta.keySet()){
			double[] tmpArray = eLogBeta.get(key);
			for(int k=0; k<K_; k++){
				LambdaSums[k] += tmpArray[k];
			}
		}
		for(int k=0; k<K_; k++){
			LambdaSums[k] += 0.001;
			LambdaSums[k] = Gamma.digamma(LambdaSums[k]);
		}
		for(int k=0; k<K_; k++){

			for(String key:eLogBeta.keySet()){
				double tmp2 = Gamma.digamma(eLogBeta.get(key)[k]) - LambdaSums[k];
				eLogBeta.get(key)[k] = tmp2;
				expELogBeta.get(key)[k] = Math.exp(tmp2);
			}
		}
	}

	private void updateEqThetas() {
		for(int d=0; d<D_; d++){
			double GammaSumK = 0;
			for(int k=0; k<K_; k++){
				GammaSumK += gamma_.get(d)[k];
			}
			
//			System.out.println("GammaSumK:" + GammaSumK);
			GammaSumK = Gamma.digamma(GammaSumK);
			
			double[] tmp = new double[K_];
			double[] expTmp = new double[K_];
			for(int k=0; k<K_; k++){
				tmp[k] = Gamma.digamma(gamma_.get(d)[k]) - GammaSumK;
				expTmp[k] = Math.exp(tmp[k]);
			}
			eLogTheta.set(d, tmp);
			expELogTheta.set(d, expTmp);
		}
	}

	private boolean changeGamma(double[] lastGamma, double[] gammaD) {
		double tmp = 0;
		for(int k=0; k<K_; k++){
			tmp += Math.abs(lastGamma[k] - gammaD[k]);
		}
		if(tmp < THRESHHOLD * K_){
			System.out.println("Diff:" + tmp);
			return true;
		}else{
			System.out.println("**Diff**:" + tmp);
			return false;
		}
	}

	private double[] copyGamma(double[] gammaD) {
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = gammaD[k];
		}
		return ret;
	}

	private void checkNewFeature(Feature[][] featureBatch) {
		// phi_
		phi_ = new ArrayList<HashMap<String, double[]>>();

		// gamma_
		gamma_ = new ArrayList<double[]>();

		// elogTheta, expElogTheta
		eLogTheta = new ArrayList<double[]>();
		expELogTheta = new ArrayList<double[]>();
		
		int allWordCount = 0;
		for(int d=0; d<D_; d++){
			allWordCount += featureBatch[d].length;
		}

		long check1start, check1end;
		long check2start, check2end;
		
		
		
		for(int d=0; d<D_; d++){
			// CALC: Gamma
			double[] tmp;
//			double[] tmpGamma = getRandomGammaArrayRegularized();
			double[] tmpGamma = getUniformalArray(1./K_);
			gamma_.add(tmpGamma);

			// CALC: elogTheta, expElogTheta
			// E[log(theta)]
			tmp = getEqlogThetaArray(tmpGamma);
			eLogTheta.add(tmp);

			// expE[log(theta)]
			double[] expTmp = tmp;
			for(int k=0; k<K_; k++){
				expTmp[k] = Math.exp(expTmp[k]);
			}
			expELogTheta.add(expTmp);
			
			// CALC: PHI, elogBeta and expElogBeta 
			HashMap<String, double[]> tmpHashMap = new HashMap<String, double[]>();  // PHI

			for(int w=0, Nd=featureBatch[d].length; w<Nd; w++){
				String tmpName = featureBatch[d][w].getName();
				
				// LAMBDA AND PHI
				if(!lambda_.containsKey(tmpName)){
					// LAMBDA 
//					tmp = getRandomGammaArrayRegularized();	
					tmp = getRandomArray();
					lambda_.put(tmpName, tmp);
					// PHI
//					tmp = getRandomGammaArrayRegularized();	// TODO confirm my decision
					tmp = getRandomGammaArray();	// TODO confirm my decision
//					for(int k=0; k<K_; k++){
//						tmp[k] /= D_;
//					}
					tmpHashMap.put(tmpName, tmp);
				}	
				
				// elogBeta expElogBeta
				if(!eLogBeta.containsKey(tmpName)){
//					check1start = System.nanoTime();
//					tmp = getEqlogEtaKW(lambda_.get(tmpName), tmpName);
					tmp = dirichlet_expectation(lambda_.get(tmpName), tmpName);
					eLogBeta.put(tmpName, tmp);
//					check1end = System.nanoTime();
//					System.out.println("CHECK1:" + (check1end - check1start));

//					check2start = System.nanoTime();
					expTmp = tmp;
					for(int k=0; k<K_; k++){
						expTmp[k] = Math.exp(tmp[k]);
					}
					expELogBeta.put(tmpName, expTmp);
//					check2end = System.nanoTime();
//					System.out.println("CHECK2:" + (check2end - check2start));
				}
			}
			phi_.add(tmpHashMap);
		}
	}
	
	private double[] getUniformalArray(double d) {
		double[] ret = new double[K_];
		Arrays.fill(ret, d);
		return ret;
	}

	private double[] getRandomArray(){
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = rnd.nextDouble() * 0.001;
		}
		return ret;
	}

	private double[] dirichlet_expectation(double[] tmpLambda, String tmpName) {	// TODO: HIGH TIME CONSUMPTION
		double[] ret = new double[K_]; 
		
		for(int k=0; k<K_; k++){
			double tmpSumLambda = 0;
			for(String key: lambda_.keySet()){
				tmpSumLambda += lambda_.get(key)[k];
			}
			ret[k] = Gamma.digamma(lambda_.get(tmpName)[k]) - Gamma.digamma(tmpSumLambda);
		}
		return ret;
	}

	private double[] getEqlogThetaArray(int d) {
		double[] ret = new double[K_]; 
		double tmpGammaSumDI = 0;
		for(int k=0; k<K_; k++){
			tmpGammaSumDI += gamma_.get(d)[k]; 
		}
		double tmpDigammadGammaSumDI = Gamma.digamma(tmpGammaSumDI);
		
		for(int k=0; k<K_; k++){
			ret[k] = Gamma.digamma(gamma_.get(d)[k]) - tmpDigammadGammaSumDI;
		}
		return ret;
	}
	private double[] getEqlogThetaArray(double[] gammaD) {
		double[] ret = new double[K_]; 
		double tmpGammaSumDI = 0;
		for(int k=0; k<K_; k++){
			tmpGammaSumDI += gammaD[k]; 
		}
		double tmpDigammadGammaSumDI = Gamma.digamma(tmpGammaSumDI);
		
		for(int k=0; k<K_; k++){
			ret[k] = Gamma.digamma(gammaD[k]) - tmpDigammadGammaSumDI;
		}
		return ret;
	}

	private double[] getRandomGammaArray() {
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = gd.sample();
		}
		return ret;
	}

	private double[] getRandomGammaArrayRegularized() {
		double[] ret = new double[K_];
		double tmp = 0;
		for(int k=0; k<K_; k++){
			ret[k] = gd.sample();
			tmp += ret[k];
		}
		
		for(int k=0; k<K_; k++){
			ret[k] /= tmp;
		}
		return ret;
	}

	@Override
	public void showTopicWords() {
		System.out.println("SHOW TOPIC WORDS:");
		System.out.println("WORD SIZE:" + lambda_.size());
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
			compareList.add(new LambdaCompare(tmpValue, word));
		}

		Collections.sort(compareList, new LambdaComparator());
		
		for(int w=0,W=compareList.size(); w<W; w++){
			ret.add(compareList.get(w).getName());
		}
		return ret;
	}	

	public static class Feature{	
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
	
	public double getPerplexity(){
		double ret = perplexity;
		perplexity = 0;
		return ret;
	}
}