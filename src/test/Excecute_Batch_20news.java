package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import ldaCore.OnlineLDA_Batch;
import ldaCore.OnlineLDA_Batch.Feature;

public class Excecute_Batch_20news {
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<Feature[][]> featureBatchList = new ArrayList<Feature[][]>();

	// Constant Parameters
	static int batchSize_ = 5;
	static String baseURI = "/Users/ishikawanaoki/Documents/workspace/LDA/targetData";
	
	// LDA Parameters
	static int K = 10;
	static double alpha = 0.1;
	static double eta = 0.1;
	static double tau0 = 0.7;
	static double kappa = 0.1;
	static int IterNum = 1;
	
	
	public static void main(String[] args) throws IOException{
		// IMPORT DATASET
		getFiles();
		
		// MAKE BATCH
		makeFeatureBatch();
		
		// generate LDA
		OnlineLDA_Batch onlineLDA_Batch = new OnlineLDA_Batch(K, alpha, eta, tau0, kappa, IterNum);
		
//		// 
//		for(int i=0; i<featureBatchList.size(); i++){
//			for(int j=0; j<featureBatchList.get(i).length; j++){
//				for(int k=0; k<featureBatchList.get(i)[j].length; k++){
//					System.out.println(featureBatchList.get(i)[j][k]);
//				}
//			}
//		}
//		// TEMP
		
		// Train
		int time = 0;
		System.out.println("T MAX:" + featureBatchList.size());
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
		String[] ret;
		line.replaceAll("\"", "");
//		line.replaceAll("\\", "");
		line.replaceAll(">", "");
		line.replaceAll("<", "");

		ret = line.split(" ");
		return ret;
	}

	private static void getFiles() {
		File parentDir = new File(baseURI);
		String[] childDirNames = parentDir.list();
		
		for(String childDirName:childDirNames){
			if(childDirName.startsWith(".")){
				continue;
			}
			String childDirURI = baseURI + "/" + childDirName;

			File childDir = new File(childDirURI);
			String[] childFileNames = childDir.list();
			
			for(String childFileName:childFileNames){
				if(childFileName.startsWith(".")){
					continue;
				}
				String childFileURI = childDirURI + "/" + childFileName;
				File childFile = new File(childFileURI);
				if(childFile.exists() && childFile!=null){
					fileNames.add(childFileURI);
				}else{
					System.out.println("File:" + childFileURI + " does not exist!");
					try {
						Thread.sleep(100000);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}
}
