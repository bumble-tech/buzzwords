[<< Back]({{ site.url }})

# Tutorial: Buzzwords Usage

## Instantiating a Model

To instantiate the model is very simple

```python
from buzzwords import Buzzwords
 
model = Buzzwords()
```

You can also change the parameters by adding your own parameter dictionary

```python
params_dict = {'UMAP': {'n_neighbors': 5}}
 
model = Buzzwords(params_dict=params_dict)
```

This will override the defaults, for more info see the API Docs

```python
model.model_parameters
 
>>> {'Embedding': {'model_name_or_path': 'paraphrase-MiniLM-L3-v2'}, 'UMAP': {'n_neighbors': 10, 'n_components': 5, 'min_dist': 0.0, 'random_state': 123}, 'HDBSCAN': {'min_cluster_size': 20, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}, 'CTFIDF': {'min_df': 1, 'num_words': 5, 'num_word_candidates': 30}, 'Buzzwords': {'similarity_threshold': 0.15, 'lemmatise_sentences': False, 'embedding_batch_size': 128}}
```
## Training a Model

To train the model on a set of documents, call the fit_transform() function to return the topics

```python
docs = df['text_column']
 
topics = model.fit_transform(docs)
```

And use the topic descriptions from the model to get the keywords for each topic

```python
first_doc_topic = topics[0]
 
model.topic_descriptions[first_doc_topic]
```

It's recommended that when using large datasets, to train the model on a sample and then predict on batches. This is to prevent memory issues as the library is _very_ memory-intensive

```python
train_docs = df.iloc[:500000]['text_column']
 
topics = model.fit_transform(train_docs)

# Reset index to prevent error from SentenceTransformer
predict_docs = df.iloc[500000:1000000]['text_column'].values.tolist()
 
topics.extend(model.transform(predict_docs))
```

## Saving/Loading a Model

Buzzwords objects offer built-in functions for saving and loading models.

```python
model = Buzzwords()

topics = model.fit_transform(df['text_column'])

model.save('models/model.buzz')
```

And similarly for loading pretrained models:

```python
model = Buzzwords()

model.load('models/model.buzz')
```

## Inference

You can use pretrained models to make inferences on new datapoints

```python
model = Buzzwords()

model.load('models/model.buzz')

topics = model.transform(df['text_column'])
```

# Image Topic Modelling

Topic modelling for images works much the same as for sentences, you just set the `model_type` to `clip` and use the paths to your images as input

```python
params_dict = {
	'Buzzwords':{
		'model_type': 'clip',
		'get_keywords': False
	},
	'Embedding': {
		'device': 'cuda',
  		'model_name_or_path': 'ViT-B/32'
	}
}

model = Buzzwords(params_dict)

# Image PATHS not image objects
image_paths = df['image_path']

topics = model.fit_transform(image_paths)
```