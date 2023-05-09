from nni.experiment import Experiment
# g和f的hidden_state
search_space = {
        'learning_rate': {'_type': 'choice', '_value': [5e-5]},
        # 'likelihoods_learning_rate': {'_type': 'choice', '_value': [1e-5, 5e-5, 1e-6]},
        'lambda_sparse':{'_type': 'choice', '_value': [5,25,50,100]},
        "warm_up_step":{'_type': 'choice', '_value': [200]},
        # "reduce_lr_step":{'_type': 'choice', '_value': [20]},
        # 'log_scale_init':{'_type': 'choice', '_value': [-1.0, -1.5, -2.0]},
        # 'pre_len': {'_type': 'choice', '_value': [96,192,336,720]},
        # 'max_steps_auglag': {'_type': 'choice', '_value': [1000]}, # config文件中目前写的2000
        # 'batch_size': {'_type': 'choice', '_value': [64]}, # config文件中目前写的64
    }

experiment = Experiment('local')
experiment.config.trial_command = 'CUDA_VISIBLE_DEVICES=4 python -m causica.run_experiment traffic --model_type rhino_informer -dc configs/dataset_config_temporal_causal_dataset_all.json --model_config configs/rhino/model_config_rhino_traffic_gaussian.json -dv gpu -i -ifc infer_config_temporal_causal_dataset.json'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
#experiment.config.use_active_gpu=True
experiment.config.max_trial_number = 400
experiment.config.trial_concurrency = 1
#experiment.config.trial_gpu_number = 3
experiment.run(8082)
#experiment.stop()
input()
