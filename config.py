import os

class Args:
    def __init__(self):
        # self.BASE_MODEL = "bert-base-uncased"
        self.BASE_MODEL = "roberta-base"
        self.BATCH_SIZE = 10
        self.SEED = 1111
        self.EPOCHS = 20
        # deprecated, no longer needed
        # self.MAX_LEN = 128
        self.task_name = "pathos_logos_all"
        #### do not change datamap_dir! it should be equal to task_name in general
        self.dir_processed_data_for_datamap = self.task_name
        self.output_train_file = os.path.join(self.dir_processed_data_for_datamap, "train.tsv")
        self.output_dev_file = os.path.join(self.dir_processed_data_for_datamap, "dev.tsv")
        self.output_test_file = os.path.join(self.dir_processed_data_for_datamap, "test.tsv")
        #### do not change lines within multiple # masks #####

        # the following lines can be changed, but I recommend using my config
        self.dir_origin_data = os.path.join(self.task_name, "origin_data")
        self.origin_train_file = os.path.join(self.dir_origin_data, "train.tsv")
        self.origin_dev_file = os.path.join(self.dir_origin_data, "dev.tsv")
        self.origin_test_file = os.path.join(self.dir_origin_data, "test.tsv")
        # whatever checkpoint path you like, datamap model will not load your model checkpoint
        self.model_path = os.path.join(self.dir_processed_data_for_datamap, "model.pt")
