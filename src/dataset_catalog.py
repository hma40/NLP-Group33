"""Dataset tasks remain separate: binary 1 has a dataset-specific meaning."""
TASKS = {
    'semeval2018': ('irony_detection', 'ironic', 'non_ironic'),
    'semeval2022': ('sarcasm_detection', 'sarcastic', 'non_sarcastic'),
    'figlang2020/twitter': ('sarcasm_detection', 'sarcastic', 'non_sarcastic'),
    'figlang2020/reddit': ('sarcasm_detection', 'sarcastic', 'non_sarcastic'),
    'sarc': ('sarcasm_detection', 'sarcastic', 'non_sarcastic'),
    'nikesh66': ('sarcasm_detection', 'sarcastic', 'non_sarcastic'),
    'hahackathon': ('humor_detection', 'humorous', 'non_humorous'),
    'semeval2022_idioms': ('idiomaticity_detection', 'idiomatic_usage', 'non_idiomatic_usage'),
}
DATASETS = list(TASKS)
PREPARATION_CHOICES = ['semeval2018', 'semeval2022', 'figlang2020', 'sarc', 'nikesh66',
                       'hahackathon', 'semeval2022_idioms']
