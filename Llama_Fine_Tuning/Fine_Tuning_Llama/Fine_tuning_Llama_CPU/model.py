from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset
from . import parameters
def train_model():
    # Load training configuration
    config = parameters.get_training_config()

    # Load dataset
    dataset = load_dataset(config['dataset_name'], split="train")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])

    # Adjust tokenizer settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=config['training_args']['output_dir'],
        num_train_epochs=config['training_args']['num_train_epochs'],
        per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training_args']['gradient_accumulation_steps'],
        save_steps=config['training_args']['save_steps'],
        logging_steps=config['training_args']['logging_steps'],
        learning_rate=config['training_args']['learning_rate'],
        weight_decay=config['training_args']['weight_decay'],
        max_grad_norm=config['training_args']['max_grad_norm'],
        warmup_ratio=config['training_args']['warmup_ratio'],
        group_by_length=config['training_args']['group_by_length'],
        lr_scheduler_type=config['training_args']['lr_scheduler_type'],
        no_cuda=True,  # Ensure that CUDA is disabled
        report_to="tensorboard"
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

# Call the training function
train_model()
