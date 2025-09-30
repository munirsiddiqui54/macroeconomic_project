# Configuration file for macroeconomic data science project

DATA_CONFIG = {
    'raw_data_path': 'data/raw',
    'interim_data_path': 'data/interim',
    'processed_data_path': 'data/processed',
    'year_range': (1971, 2024)
}

MODEL_CONFIG = {
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'test_size': 0.2
}

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
