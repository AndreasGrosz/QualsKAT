# 240802 Train log.txt

AAUNaL\**python3 main.py --train**

Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-base-v2 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of DebertaForSequenceClassification were not initialized from the model checkpoint at microsoft/deberta-base and are newly initialized: ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

### Verarbeite Modell: distilbert-base-uncased

Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|████████████████████████████████████████| 1735/1735 [00:14<00:00, 120.39 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:03<00:00, 94.85 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:05<00:00, 64.73 examples/s]
/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
  1%|▌                                                      | 3/327 [00:57<1:42:49, 19.04s/it^{'eval_loss': 0.7685518860816956, 'eval_accuracy': 0.8629032373428345, 'eval_runtime': 122.8256, 'eval_samples_per_second': 3.029, 'eval_steps_per_second': 0.195, 'epoch': 1.0}
{'eval_loss': 0.47202619910240173, 'eval_accuracy': 0.9059139490127563, 'eval_runtime': 123.2846, 'eval_samples_per_second': 3.017, 'eval_steps_per_second': 0.195, 'epoch': 2.0}
{'eval_loss': 0.4218517541885376, 'eval_accuracy': 0.9032257795333862, 'eval_runtime': 121.37, 'eval_samples_per_second': 3.065, 'eval_steps_per_second': 0.198, 'epoch': 3.0}
{'train_runtime': 6217.8269, 'train_samples_per_second': 0.837, 'train_steps_per_second': 0.053, 'train_loss': 1.018005744397458, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████| 327/327 [1:43:37<00:00, 19.01s/it]
100%|█████████████████████████████████████████████████████████| 24/24 [01:58<00:00,  4.92s/it]

### Verarbeite Modell: roberta-base

Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|████████████████████████████████████████| 1735/1735 [00:16<00:00, 103.58 examples/s]
Map: 100%|██████████████████████████████████████████| 372/372 [00:03<00:00, 111.56 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:05<00:00, 73.92 examples/s]
/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
{'eval_loss': 0.5054287910461426, 'eval_accuracy': 0.897849440574646, 'eval_runtime': 249.9704, 'eval_samples_per_second': 1.488, 'eval_steps_per_second': 0.096, 'epoch': 1.0}
{'eval_loss': 0.4260542690753937, 'eval_accuracy': 0.9032257795333862, 'eval_runtime': 250.6052, 'eval_samples_per_second': 1.484, 'eval_steps_per_second': 0.096, 'epoch': 2.0}
{'eval_loss': 0.3954486548900604, 'eval_accuracy': 0.9086021780967712, 'eval_runtime': 258.8797, 'eval_samples_per_second': 1.437, 'eval_steps_per_second': 0.093, 'epoch': 3.0}
{'train_runtime': 12963.9756, 'train_samples_per_second': 0.401, 'train_steps_per_second': 0.025, 'train_loss': 0.8307503516520929, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████| 327/327 [3:36:03<00:00, 39.65s/it]
100%|█████████████████████████████████████████████████████████| 24/24 [03:59<00:00, 10.00s/it]

### Verarbeite Modell: albert-base-v2

Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-base-v2 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|█████████████████████████████████████████| 1735/1735 [00:22<00:00, 77.87 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:04<00:00, 82.43 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:06<00:00, 57.80 examples/s]
/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
{'eval_loss': 0.5463130474090576, 'eval_accuracy': 0.8870967626571655, 'eval_runtime': 270.6537, 'eval_samples_per_second': 1.374, 'eval_steps_per_second': 0.089, 'epoch': 1.0}
{'eval_loss': 0.4370790719985962, 'eval_accuracy': 0.897849440574646, 'eval_runtime': 270.5456, 'eval_samples_per_second': 1.375, 'eval_steps_per_second': 0.089, 'epoch': 2.0}
{'eval_loss': 0.4041113555431366, 'eval_accuracy': 0.9032257795333862, 'eval_runtime': 268.0647, 'eval_samples_per_second': 1.388, 'eval_steps_per_second': 0.09, 'epoch': 3.0}
{'train_runtime': 13022.8814, 'train_samples_per_second': 0.4, 'train_steps_per_second': 0.025, 'train_loss': 0.7305391644119122, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████| 327/327 [3:37:02<00:00, 39.83s/it]
100%|█████████████████████████████████████████████████████████| 24/24 [04:18<00:00, 10.77s/it]

### Verarbeite Modell: microsoft/deberta-base

Some weights of DebertaForSequenceClassification were not initialized from the model checkpoint at microsoft/deberta-base and are newly initialized: ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map:   0%|                                                    | 0/1735 [00:00<?, ? examples/s]Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no padding.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Map: 100%|█████████████████████████████████████████| 1735/1735 [00:23<00:00, 74.05 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:04<00:00, 80.42 examples/s]
Map: 100%|███████████████████████████████████████████| 372/372 [00:05<00:00, 62.14 examples/s]
/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
  0%|                                                                 | 0/327 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/main.py", line 119, in <module>
    main()
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/main.py", line 78, in main
    trainer.train()
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/trainer.py", line 1938, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/trainer.py", line 2279, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/trainer.py", line 3318, in training_step
    loss = self.compute_loss(model, inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/trainer.py", line 3363, in compute_loss
    outputs = model(**inputs)
              ^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/models/deberta/modeling_deberta.py", line 1196, in forward
    outputs = self.deberta(
              ^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/models/deberta/modeling_deberta.py", line 964, in forward
    encoder_outputs = self.encoder(
                      ^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/models/deberta/modeling_deberta.py", line 460, in forward
    hidden_states = layer_module(
                    ^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/models/deberta/modeling_deberta.py", line 373, in forward
    attention_output = self.attention(
                       ^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/models/deberta/modeling_deberta.py", line 306, in forward
    self_output = self.self(
                  ^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/res/HDD_AB/projekte/kd0241-py/Q-KAT/AAUNaL/venv/lib/python3.12/site-packages/transformers/models/deberta/modeling_deberta.py", line 648, in forward
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 237949750272 bytes. Error code 12 (

**Cannot allocate memory)**
  0%|

