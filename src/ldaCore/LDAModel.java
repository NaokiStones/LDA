package ldaCore;

import ldaCore.OnlineLDA_Batch.Feature;

public interface LDAModel {

	void trainBatch(Feature[][] featureBatch, int time);

	void showTopicWords();

}
