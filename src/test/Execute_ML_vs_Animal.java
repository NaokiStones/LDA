package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;


import ldaCore.OnlineLDA_Batch;
import ldaCore.OnlineLDA_Batch.Feature;

public class Execute_ML_vs_Animal{
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<Feature[][]> featureBatchList = new ArrayList<Feature[][]>();

	// Constant Parameters
	static int batchSize_ = 1;
	static String baseURI;
	
	// limit Read File Per Directory
	static int limit = 100;
	
	// LDA Parameters
	static int K = 2;
	static double alpha = 1./K;
	static double eta = 1./ K;
	static double tau0 = 1024;
	static double kappa = 0.7;
	static int IterNum = 200;
	
	
	public static void main(String[] args) throws IOException{
		// Set Directory
		baseURI = "/Users/ishikawanaoki/Documents/workspace/LDA/myCorpus/";
		// IMPORT DATASET
		getFiles(baseURI);
		
		// MAKE BATCH
		makeFeatureBatch();
		
		// generate LDA
		OnlineLDA_Batch onlineLDA_Batch = new OnlineLDA_Batch(K, alpha, eta, tau0, kappa, IterNum);
		
		// 
//		for(int i=0; i<featureBatchList.size(); i++){
//			for(int j=0; j<featureBatchList.get(i).length; j++){
//				for(int k=0; k<featureBatchList.get(i)[j].length; k++){
//					System.out.println(featureBatchList.get(i)[j][k]);
//				}
//			}
//		}
		// TEMP
		// Train *************************************************
		int time = 0;
		System.out.println("T MAX:" + featureBatchList.size() * batchSize_);
		for(Feature[][] featureBATCH:featureBatchList){
			System.out.println("time:" + time);
			onlineLDA_Batch.trainBatch(featureBATCH, time);
			time += batchSize_;
		}
		
		// Output
		System.out.println("start Output");
		onlineLDA_Batch.showTopicWords();
		System.out.println("end Output");
	}
	
	private static void makeFeatureBatch() throws IOException {
		for(int batchIdx=0, BATCHSIZE=fileNames.size(); batchIdx < BATCHSIZE; batchIdx+=batchSize_){
			int tmpBatchSize = -1;
			if(batchIdx + batchSize_ >= BATCHSIZE){
				tmpBatchSize = BATCHSIZE - batchIdx;
			}else{
				tmpBatchSize = batchSize_;
			}
			
			Feature[][] tmpFeatureBatch = new Feature[tmpBatchSize][];
			
			for(int tmpLocalBatchIdx=0; tmpLocalBatchIdx<tmpBatchSize; tmpLocalBatchIdx++){
				int tmpBatchIdx = batchIdx + tmpLocalBatchIdx;
				tmpFeatureBatch[tmpLocalBatchIdx] = makeFeature(tmpBatchIdx);
			}
			featureBatchList.add(tmpFeatureBatch);
		}
	}
	
	private static Feature[] makeFeature(int tmpBatchIdx) throws IOException {
		Feature[] ret;
		HashMap<String, Integer> tmpHashMap = new HashMap<String, Integer>(); 
		
		String targetFileURI = fileNames.get(tmpBatchIdx);
		
		BufferedReader br;
		try{
			br = new BufferedReader(new FileReader(targetFileURI));
		}catch(FileNotFoundException e){
			System.out.println("targetFileURI:" + targetFileURI);
			throw new FileNotFoundException();
		}
		
		
		try{
			while(true){
				String line = br.readLine();
				if(line==null)
					break;
				String[] words = processLine(line);
				for(String word: words){
					if(tmpHashMap.containsKey(word)){
						int nextNum = tmpHashMap.get(word) + 1;
						tmpHashMap.put(word, nextNum);
					}else{
						tmpHashMap.put(word, 1);
					}
				}
			}
		}finally{
			try {
				br.close();
			} catch (IOException e) {
				System.out.println("br close miss");
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
			}
			System.out.println(tmpHashMap.size());
		}
		
		for (Iterator<String> i = tmpHashMap.keySet().iterator(); i.hasNext();) {
			if (i.next().equals(1)) {
				i.remove();
			}
		}


		int tmpHashMapSize = tmpHashMap.size();
		ret = new Feature[tmpHashMapSize];
		
		int tmpIdx = 0;
		for(String key: tmpHashMap.keySet()){
			int tmpCount = -1;
			tmpCount = tmpHashMap.get(key);
			ret[tmpIdx] = new Feature(key, tmpCount);	// TODO OK?
			tmpIdx++;
//			System.out.println("ret[tmpIdx]" + ret[tmpIdx].getName() + " " + ret[tmpIdx].getCount());
		}
		

		return ret;
	}

	private static String[] processLine(String line) {
		
		line = line.toLowerCase();
		
		line = line.replace("\"", " ");
		line = line.replace("\\", " ");
		line = line.replace("/", " ");
		line = line.replace(">", " ");
		line = line.replace("<", " ");
		line = line.replace("-", " ");
		line = line.replace(",", " ");
		line = line.replace(".", " ");
		line = line.replace("(", " ");
		line = line.replace(")", " ");
		line = line.replace(":", " ");
		line = line.replace(";", " ");
		line = line.replace("'", " ");
		line = line.replace("[", " ");
		line = line.replace("]", " ");
		line = line.replace("!", " ");
		line = line.replace("*", " ");
		line = line.replace("#", " ");
		line = line.replace("+", " ");
		line = line.replace("%", " ");
		line = line.replace("@", " ");
		line = line.replace("&", " ");
		line = line.replace("?", " ");
		line = line.replace("$", " ");
		line = line.replace("0", " ");
		line = line.replace("1", " ");
		line = line.replace("2", " ");
		line = line.replace("3", " ");
		line = line.replace("4", " ");
		line = line.replace("5", " ");
		line = line.replace("6", " ");
		line = line.replace("7", " ");
		line = line.replace("8", " ");
		line = line.replace("9", " ");
		line = line.replace("\t", " ");
		line = line.replace("_", " ");
		line = line.replace("{", " ");
		line = line.replace("}", " ");
		line = line.replace("=", " ");
		line = line.replace("|", " ");
		
		
		String[] ret;
		
		ret = line.split(" ");
		
		
		for(int i=0; i<ret.length; i++){
			ret[i] = ret[i].replace(" ", "");
		}

		return ret;
	}

	private static void getFiles(String str) {
		File dir = new File(str);
		String[] files = dir.list();
		for(String tmpFileName: files){
			String tmpURI = str + tmpFileName;
			fileNames.add(tmpURI);
		}
	}
}
