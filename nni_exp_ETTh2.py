# data_config 后缀 ett
# 命令待改


from nni.experiment import Experiment

search_space = {
        'learning_rate': {'_type': 'choice', '_value': [1e-4, 5e-4, 1e-3]},
        'likelihoods_learning_rate': {'_type': 'choice', '_value': [1e-4, 5e-4, 1e-3]},
        'max_steps_auglag': {'_type': 'choice', '_value': [1,2,5]},
        'max_auglag_inner_epochs': {'_type': 'choice', '_value': [2000, 6000, 20000]},
        'lambda_sparse':{'_type': 'choice', '_value': [100,25,200,50]},
        'log_scale_init':{'_type': 'choice', '_value': [0.0, -1.0,-5.0]},
    }
        #'pre_len': {'_type': 'choice', '_value': [96,192,336,720]},
        # 'learning_rate': {'_type': 'choice', '_value': [0.0001, 0.001, 0.01]},
        # 'likelihoods_learning_rate': {'_type': 'choice', '_value': [0.0001, 0.001, 0.01]},
        # 'max_steps_auglag': {'_type': 'choice', '_value': [5,10,30,50]},
        # 'max_auglag_inner_epochs': {'_type': 'choice', '_value': [2000,4000,6000]},
        # # 'tau_gumbel': {'_type': 'choice', '_value': [0.1,0.2,0.3,0.4]},
        # #'pre_len': {'_type': 'choice', '_value': [96,192,336,720]},
experiment = Experiment('local')
experiment.config.trial_command = 'CUDA_VISIBLE_DEVICES=6 python -m causica.run_experiment weather --model_type rhino -dc configs/dataset_config_temporal_causal_dataset_ett.json --model_config configs/rhino/model_config_rhino_weather_gaussian.json -dv gpu -i -ifc infer_config_temporal_causal_dataset.json'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
#experiment.config.use_active_gpu=True
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 1
#experiment.config.trial_gpu_number = 3
experiment.run(8087)
#experiment.stop()
