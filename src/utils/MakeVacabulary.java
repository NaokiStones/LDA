package utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class MakeVacabulary {
	private static int limit = 100;
	private static ArrayList<String> fileNames = new ArrayList<String>();
	private static Set<String> set = new HashSet<String>();
	private static FileWriter filewriter;
	public static void main(String args[]){
		String SourcePath = "/Users/ishikawanaoki/Documents/workspace/LDA/targetData";
		String OutputPath = "/Users/ishikawanaoki/Documents/workspace/LDA/vocab/vocabulary.txt";
		
		File file = new File(OutputPath);
		try {
			filewriter = new FileWriter(file, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		

		getFiles(SourcePath);
		makeVocabularyList();
		
	}
	private static void makeVocabularyList() {
		for(String path:fileNames){

			BufferedReader reader = null;
			try {
				reader = new BufferedReader(
					    new InputStreamReader(new FileInputStream(path), "UTF-8"));
			} catch (UnsupportedEncodingException e1) {
				e1.printStackTrace();
			} catch (FileNotFoundException e1) {
				e1.printStackTrace();
			}		
			while(true){
				String tmpStr = null;
				try {
					tmpStr = reader.readLine();
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				if(tmpStr==null){
					break;
				}
				
				getWords(tmpStr);
			}
		}
		for(String word:set){
			if(word.length() == 0) continue;
			try {
				filewriter.write(word + "\n");
			} catch (IOException e) {
				e.printStackTrace();
			} 
		}
	}
	private static void getWords(String line) {
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
		line = line.replace("\n", " ");
		
		String[] tmpWords = line.split(" ");
		
		for(String word:tmpWords){
			if(!set.contains(word)){
				set.add(word);
			}
		}
	}
	private static void getFiles(String sourcePath) {
		File parentDir = new File(sourcePath);
		String[] childDirNames = parentDir.list();
		
		for(String childDirName:childDirNames){
			if(childDirName.startsWith(".")){
				continue;
			}
			String childDirURI = sourcePath + "/" + childDirName;

			File childDir = new File(childDirURI);
			String[] childFileNames = childDir.list();
			
			int cnt = 0;
			for(String childFileName:childFileNames){
				if(limit != -1){
					if(cnt >= limit)
						continue;
				}
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
