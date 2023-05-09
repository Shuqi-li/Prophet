from nni.experiment import Experiment
# g和f的hidden_state
search_space = {
        'learning_rate': {'_type': 'choice', '_value': [1e-4, 1e-5, 1e-6]},
        # 'likelihoods_learning_rate': {'_type': 'choice', '_value': [1e-5, 5e-5, 1e-6]},
        'lambda_sparse':{'_type': 'choice', '_value': [0,5,100]},
        "warm_up_step":{'_type': 'choice', '_value': [25,50]},
        # "reduce_lr_step":{'_type': 'choice', '_value': [20]},
        'log_scale_init':{'_type': 'choice', '_value': [-1.76,-1.519,-1.273,-1.064]},
        'pre_len': {'_type': 'choice', '_value': [96,192,336,720]},
        # 'pre_len': {'_type': 'choice', '_value': [192,336,720]},
        'max_steps_auglag': {'_type': 'choice', '_value': [50]},
        'batch_size': {'_type': 'choice', '_value': [128]}, # config文件中目前写的64
        # 'random_seed': {'_type': 'choice', '_value': [1,2,3,4,5,6]},
    }


experiment = Experiment('local')
experiment.config.trial_command = 'CUDA_VISIBLE_DEVICES=2 python -m causica.run_experiment weather --model_type rhino_informer -dc configs/dataset_config_temporal_causal_dataset_all.json --model_config configs/rhino/model_config_rhino_weather_gaussian.json -dv gpu -i -ifc infer_config_temporal_causal_dataset.json'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
#experiment.config.use_active_gpu=True
experiment.config.max_trial_number = 288
experiment.config.trial_concurrency = 3
#experiment.config.trial_gpu_number = 3
experiment.run(8093)
#experiment.stop()
input()
