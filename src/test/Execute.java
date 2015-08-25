package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import ldaCore.OnlineLDA;
import ldaCore.OnlineLDA.Feature;

public class Execute {
	
	private static final int tmpK = 30;
	private static final double tmpTau0 = 0.1;
	private static final double tmpKappa= 0.7;	// (0.5, 1.0]
	private static final double tmpEta0 = 0.1;
	private static final double tmpAlpha= 0.3;

	public static void main(String[] args) {
		// IMPORT FILE
		ArrayList<Feature[]> featureArrayList = new ArrayList<Feature[]>();
		File file = new File("/Users/ishikawanaoki/Documents/datasetML/unlabeledTrainData.tsv");
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		String str;
		try {
			while((str = br.readLine()) != null){
				String[] words = str.split("\t");
				words[1].replaceAll(",", "");
				words[1].replaceAll(".", "");
				words[1].replaceAll("\"", "");
				
				String[] sentence = words[1].split(" ");
				
				int N = sentence.length;
				HashMap<String, Integer> hashMap = new HashMap<String, Integer>();
				for(int i=0; i<N; i++){
					String tmpWord = sentence[i];
					if(!hashMap.containsKey(tmpWord)){
						hashMap.put(tmpWord, 1);
					}else{
						int nextNum = hashMap.get(tmpWord) + 1;
						hashMap.put(tmpWord, nextNum);
					}
				}
				int hashMapSize = hashMap.size();
				Feature[] tmpFeature = new Feature[hashMapSize];
				int tmpIdx = 0;
				for(String tmpStr:hashMap.keySet()){
					tmpFeature[tmpIdx] = new Feature(tmpStr, hashMap.get(tmpStr));
//					System.out.print(tmpStr + ", " + hashMap.get(tmpStr) + " ");
				}
//				System.out.println("");
				featureArrayList.add(tmpFeature);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		// ONLINE LDA instance
		OnlineLDA onlineLDA = new OnlineLDA(tmpK, tmpTau0, tmpKappa, tmpEta0, tmpAlpha);

		// TEST
		for(int i=0,SIZE=featureArrayList.size(); i<SIZE; i++){
			System.out.println("#" + i);
			onlineLDA.trainPerLine(featureArrayList.get(i), i);	// features, time
		}

		// PRINT RESULT

	}

}
