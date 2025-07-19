"""
ClearML Hyperparameter Optimization Script for Enhanced CNN Training
This script demonstrates how to use ClearML for automated hyperparameter optimization
"""

from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import DiscreteParameterRange, LogUniformParameterRange
import subprocess
import sys
import os

def create_optimization_task():
    """Create a ClearML task for hyperparameter optimization"""
    
    # Initialize the optimization task
    task = Task.init(
        project_name='CNN-Hyperparameter-Optimization',
        task_name='ResNet-AutoML-Optimization',
        task_type=Task.TaskTypes.optimizer
    )
    
    # Define hyperparameter search space
    optimization_params = {
        # Learning rate optimization
        'Args/lr': LogUniformParameterRange(min_value=1e-5, max_value=1e-2),
        
        # Weight decay optimization
        'Args/weight_decay': LogUniformParameterRange(min_value=1e-6, max_value=1e-3),
        
        # Batch size optimization
        'Args/batch': DiscreteParameterRange([16, 32, 48, 64]),
        
        # Scheduler type optimization
        'Args/scheduler': DiscreteParameterRange(['plateau', 'cosine', 'onecycle']),
        
        # Model architecture optimization
        'Args/model_name': DiscreteParameterRange(['resnet50', 'resnet101', 'resnet152']),
        
        # Image size optimization
        'Args/image_size': DiscreteParameterRange([192, 224, 256, 288]),
        
        # Training epochs
        'Args/epochs': UniformIntegerParameterRange(min_value=30, max_value=100),
        
        # Patience for early stopping
        'Args/patience': UniformIntegerParameterRange(min_value=5, max_value=20)
    }
    
    # Create the optimizer
    optimizer = HyperParameterOptimizer(
        base_task_id=None,  # Will be set to template task
        hyper_parameters=optimization_params,
        objective_metric_title='Val_Metrics',
        objective_metric_series='F1_Score',
        objective_metric_sign='max',  # Maximize F1 score
        max_number_of_concurrent_tasks=2,  # Adjust based on available resources
        optimizer_class='OptimizerBOHB',  # Bayesian optimization
        execution_queue='default',
        max_iteration_per_job=100,
        total_max_jobs=50,
        min_iteration_per_job=10,
        save_top_k_tasks_only=10
    )
    
    return optimizer, task

def create_template_task():
    """Create a template task for the optimization"""
    
    template_task = Task.init(
        project_name='CNN-Hyperparameter-Optimization',
        task_name='ResNet-Template-Task',
        reuse_last_task_id=False
    )
    
    # Set template parameters (these will be overridden by optimizer)
    template_args = {
        'data_dir': 'data_dir',
        'image_size': 224,
        'workers': 6,
        'batch': 32,
        'model_name': 'resnet152',
        'pretrained': True,
        'epochs': 50,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'scheduler': 'cosine',
        'patience': 10,
        'project': 'runs/optimization',
        'name': 'optimized_run',
        'use_weighted_sampling': True,
        'stream_artifacts': True
    }
    
    template_task.connect(template_args)
    
    # Set the script to execute
    template_task.set_script(
        script_path='train_local_sonnet.py',
        working_directory='.',
        # Use uv run for execution
        python_binary='uv'
    )
    
    # Mark as template and close
    template_task.mark_started()
    template_task.mark_stopped()
    
    return template_task.id

def run_hyperparameter_optimization():
    """Run the complete hyperparameter optimization pipeline"""
    
    print("🚀 Starting ClearML Hyperparameter Optimization for Enhanced CNN Training")
    print("=" * 80)
    
    # Create template task
    print("📋 Creating template task...")
    template_task_id = create_template_task()
    print(f"✅ Template task created: {template_task_id}")
    
    # Create optimization task
    print("🔧 Setting up hyperparameter optimizer...")
    optimizer, opt_task = create_optimization_task()
    
    # Set the template task
    optimizer.set_base_task_id(template_task_id)
    
    print("🎯 Optimization configuration:")
    print(f"   - Objective: Maximize Validation F1 Score")
    print(f"   - Max concurrent tasks: 2")
    print(f"   - Total max jobs: 50")
    print(f"   - Optimizer: Bayesian Optimization (BOHB)")
    print(f"   - Save top: 10 tasks")
    
    print("\n🔍 Search space:")
    print(f"   - Learning Rate: [1e-5, 1e-2] (log-uniform)")
    print(f"   - Weight Decay: [1e-6, 1e-3] (log-uniform)")
    print(f"   - Batch Size: [16, 32, 48, 64]")
    print(f"   - Scheduler: [plateau, cosine, onecycle]")
    print(f"   - Model: [resnet50, resnet101, resnet152]")
    print(f"   - Image Size: [192, 224, 256, 288]")
    print(f"   - Epochs: [30, 100]")
    print(f"   - Patience: [5, 20]")
    
    print("\n🎬 Starting optimization...")
    print("💡 Monitor progress in ClearML WebUI")
    print("🔗 View results at: https://app.clear.ml")
    
    # Start the optimization
    optimizer.start()
    
    # Wait for completion
    optimizer.wait()
    
    # Get best parameters
    best_task = optimizer.get_top_experiments(top_k=1)[0]
    print(f"\n🏆 Optimization completed!")
    print(f"🥇 Best task ID: {best_task.id}")
    print(f"📊 Best F1 Score: {best_task.get_last_scalar_metrics()['Val_Metrics']['F1_Score']}")
    
    # Print best hyperparameters
    best_params = best_task.get_parameters()
    print(f"\n🎯 Best hyperparameters:")
    for section, params in best_params.items():
        if section == 'Args':
            for param, value in params.items():
                print(f"   - {param}: {value}")
    
    print(f"\n✨ Optimization results saved to ClearML project: CNN-Hyperparameter-Optimization")
    print(f"🔧 Use these parameters for your production training!")
    
    return best_task, optimizer

def generate_best_config(best_task):
    """Generate a configuration file with the best parameters"""
    
    best_params = best_task.get_parameters()['Args']
    
    config_content = f"""# Best Hyperparameters from ClearML Optimization
# Task ID: {best_task.id}
# F1 Score: {best_task.get_last_scalar_metrics()['Val_Metrics']['F1_Score']:.4f}

# Optimized Training Command:
uv run train_local_sonnet.py \\
    --lr {best_params.get('lr', 0.001)} \\
    --weight-decay {best_params.get('weight_decay', 0.0001)} \\
    --batch {best_params.get('batch', 32)} \\
    --scheduler {best_params.get('scheduler', 'cosine')} \\
    --model-name {best_params.get('model_name', 'resnet152')} \\
    --image-size {best_params.get('image_size', 224)} \\
    --epochs {best_params.get('epochs', 50)} \\
    --patience {best_params.get('patience', 10)} \\
    --use-weighted-sampling \\
    --stream-artifacts

# Individual Parameters:
"""
    
    for param, value in best_params.items():
        config_content += f"# {param}: {value}\n"
    
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(config_content)
    
    print(f"💾 Best configuration saved to: best_hyperparameters.txt")

if __name__ == "__main__":
    try:
        # Run optimization
        best_task, optimizer = run_hyperparameter_optimization()
        
        # Generate configuration file
        generate_best_config(best_task)
        
        print(f"\n🎉 Hyperparameter optimization completed successfully!")
        print(f"📈 Check the ClearML WebUI for detailed results and comparisons")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Optimization interrupted by user")
        print(f"🔄 You can resume the optimization later through ClearML WebUI")
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print(f"🔧 Check your ClearML configuration and try again")
    
    print(f"\n🏁 Script completed!")
