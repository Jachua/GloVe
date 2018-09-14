## GloVe: SVM classifier with word2vec embeddings


Train an SVM classifier on labeled documents with supporting vectors generated from word embeddings. Classify text labels on test entries.

```
$ ./train.sh path/to/corpus
$ ./run.sh --model_file corpus --test_file path/to/test --sample_length 30 --sample_step 20 --kernel_type linear --C_value 100
```


Origin repo [here](https://github.com/stanfordnlp/GloVe)