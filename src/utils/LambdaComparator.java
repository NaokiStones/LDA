package utils;

import java.util.Comparator;

public class LambdaComparator implements Comparator<LambdaCompare>{

	@Override
	public int compare(LambdaCompare o1, LambdaCompare o2) {
		double v1 = o1.getValue();
		double v2 = o2.getValue();
		
		if(v1 < v2){
			return 1;
		}else if(v2 < v1){
			return -1;
		}else {
			return 0;
		}
	}

}
