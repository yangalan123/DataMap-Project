# DataMap Pipeline for Active Learning and Data Verification
Author: Chenghao Yang (yangalan1996@gmail.com)
## Requirements (include those from DataMap)
- loguru == 0.3.0
- transformers == 4.0.1
- torch == 1.7.0
- tqdm == 4.43.0
- seaborn == 0.10.0
- pandas == 1.0.3
- matplotlib == 3.2.1
- numpy == 1.18.2
- jsonnet == 0.15.0
- tensorboardx == 2.0
- torch == 1.4.0
- spacy == 2.1.9
- scikit-learn == 0.22.2.post1

## How to Use
1. Read `config.py` and sample data in TestTask. Prepare your data as the same format as the sample data and doing necessary edits to `config.py`. Generally, you can play with the model hyper-parameter, seed, and the task name. Feel free to rename `TestTask/` to other names like `TASK_NAME/`, as long as you make corresponding changes in `config.py`.
1. Run `python main.py` to train the model. 
1. Run `python data_processor.py` to do necessary post-editing for data files generated in `TASK_NAME/`. Then run `python log_processor.py` to prepare the training dynamics data.
1. Switch to `cartography/` by the command `cd cartography`.
1. Run `python -m cartography.selection.train_dy_filtering --task_name TASK_NAME --data_dir ../ --plots_dir ../datamap_figures --model_dir ../ --filter --metric Metrics --plot` to plot the datamap figure and filter hard-to-learn / ambiguous data. For more  details about this command, please read README.md in `cartography/`.

