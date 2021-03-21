# Arabic-socialMedia-evaluation
An **Arabic** language-based Sentiment detection and objective topics categorization system.

**NOTE:** The function use case is explained in the [`main.ipynb`](./main.ipynb) file.


*	In order to run this module, you should install the packages in [`requirements.txt`](./requirements.txt).
	Enter the following command in your terminal: 
```bash
pip install -r requirements.txt
```
*	This module takes a dataframe of text entries in Arabic (Egypt and KSA dialects) and predicts their Sentiment whether positive, negative or objective.
*	The objective class represents the non-feedback category, so the second task of the module is to categorize the different topics within the objective class and find common inquiries in users feedback.
*	This module uses four different models for class evaluation, previously trained over hundreds of thousands of Egyptian and Saudi social media interactions and reviews collected from several resources.
*	The pre-trained models were constructed with different features including the text vectorizing methods and the architecture.
*	The first model is a fully connected neural network that uses Tf-Idf vectorization; [`FCNN_tfidf_model`](./models/FCNN_tfidf_model).
*	The second model is a fully connected neural network that uses Word2Vec vectorization that was build from scratch on Arabic and English documents using skip-gram training; [`FCNN_w2v_model`](./models/FCNN_w2v_model).
*	The third is a bidirectional LSTM-based neural network with Word2Vec vectorization; [`FCNN_w2v_lstm_model`](./models/FCNN_w2v_lstm_model).
*	The last model is a simple Lexicon based prediction model.

You can download the latest version of the word2vec model from this [link](https://drive.google.com/file/d/1ak7QjRZ0GcFbS-BzCbQU1mkrQEEx_HPr)

To load the w2v model:
```python
from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format("./models/w2v_model.bin", binary=True)
```

For more support and info, send an email to: mina.melek@andalusiagroup.net
