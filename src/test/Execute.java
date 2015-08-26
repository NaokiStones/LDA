package test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
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
//		// IMPORT FILE
//		ArrayList<Feature[]> featureArrayList = new ArrayList<Feature[]>();
//		
//		String basePath = "";
//		featureArrayList = getFeatureArrayListFromDocuments(basePath);
//		
//
//		// ONLINE LDA instance
//		OnlineLDA onlineLDA = new OnlineLDA(tmpK, tmpTau0, tmpKappa, tmpEta0, tmpAlpha);
//
//		// TEST
//		for(int i=0,SIZE=featureArrayList.size(); i<SIZE; i++){
//			System.out.println("#" + i);
//			onlineLDA.trainPerLine(featureArrayList.get(i), i);	// features, time
//		}
//
//		// PRINT RESULT

	}

//	private static ArrayList<Feature[]> getFeatureArrayListFromDocuments(String basePath) {
//		ArrayList<Feature[]> ret = new ArrayList<Feature[]>();
//		File dir = new File(basePath);
//		File[] files = getFiles(dir);
//		
//		for(File file:files){
//			ret.add()
//		}
//		
//		return ret;
//	}

}
