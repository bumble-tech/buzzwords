from buzzwords import Buzzwords

DEFAULT_PARAMS = {
    'Embedding': {
        'model_name_or_path': 'paraphrase-multilingual-mpnet-base-v2'
    },
    'UMAP': {
        'n_neighbors': 10,
        'n_components': 5,
        'min_dist': 0.0,
        'random_state': 123,
        'n_epochs': 500,
    },
    'HDBSCAN': {
      'min_cluster_size': 10,
      'min_samples': 15,
      'metric': 'euclidean',
      'cluster_selection_method': 'eom',
    },
    'Keywords': {
        'min_df': 1,
        'num_words': 10,
        'num_word_candidates': 15,
    },
    'Buzzwords': {
        'lemmatise_sentences': False,
        'get_keywords': True,
        'embedding_batch_size': 128,
        'matryoshka_decay': 0.8,
        'keyword_backend': 'keybert',
        'prediction_data': True,
        'model_type': 'sentencetransformers'
    }
}

def test_custom_input():
    params = {
        'UMAP': {
            'n_neighbours': 200
        },
        'HDBSCAN': {
            'min_cluster_size': 2
        },
        'Keywords': {
            'min_df': 100
        },
        'Buzzwords': {
            'matryoshka_decay': DEFAULT_PARAMS['Buzzwords']['matryoshka_decay'] - 0.01
        }
    }
    
    model = Buzzwords(params)

    assert model.model_parameters['UMAP']['n_neighbours'] == 200
    assert model.model_parameters['HDBSCAN']['min_cluster_size'] == 2
    assert model.model_parameters['Keywords']['min_df'] == 100
    assert model.matryoshka_decay == DEFAULT_PARAMS['Buzzwords']['matryoshka_decay'] - 0.01

    del model

def test_model_fit():
    model = Buzzwords(DEFAULT_PARAMS)

    text = ['test', 'haha', 'ok']*100

    topics = model.fit_transform(text)

    assert len(topics) == len(text)

    del model

def test_model_save_load(tmp_path):
    model = Buzzwords(DEFAULT_PARAMS)

    text = ['test', 'haha', 'ok']*1000

    model.fit_transform(text)

    topics = model.transform(text)

    model.save(tmp_path)

    del model

    model = Buzzwords()

    model.load(tmp_path)

    topics_loaded = model.transform(text)

    assert len(topics_loaded) == len(text)
    assert (topics == topics_loaded).all()

    del model

def test_recursions():
    updated_params = DEFAULT_PARAMS

    updated_params['UMAP'] = {
        'n_neighbors': 1,
        'n_components': 25,
        'min_dist': 0.0,
        'random_state': 123,
        'n_epochs': 500,
    }

    updated_params['HDBSCAN'] = {
        'min_cluster_size': 800,
        'min_samples': 5,
    }
    
    model = Buzzwords(updated_params)

    text = ['football']*1000 + ['dsadsidlasda', 'test', 'ok']

    topics = model.fit_transform(
        text,
        recursions=2
    )

    assert len(model.umap_models) == 2
    assert len(model.hdbscan_models) == 2
    del model

def test_topic_merge():
    model = Buzzwords(DEFAULT_PARAMS)

    text = ['test', 'haha', 'ok']*1000

    topics = model.fit_transform(text)

    model.merge_topic(topics[0], topics[0]+10000)

    topics_redone = model.transform(text)

    assert topics[0]+10000 in topics_redone
    assert topics[0] not in topics_redone
    del model
