### run a simple example:
    ##### prepare:
        unzip files:
            ./MANSION/data/test/ruby/
            ./MANSION/data/train_valid/ruby/
        download a model:
            /java_fine_turn_GraBertlastepoch_lookLoss_GraBert/download_info.txt

    ##### train a knowledge module：
        1, train a ruby knowledge module
            train: run src/train_kms/train_kms.py
            infer: run src/train_kms/kms_inference.py
            get mrr: run src/train_kms/mrr.py

    ##### train a knowledge fusion module:
        2, train a ruby knowledge fusion module
            train: run src/train_fusion/fusion_final_version.py
            infer: run src/train_fusion/fusion_inference.py
            get mrr: run src/train_kms/mrr.py



### File structure:
        /data: training, testing, validation datasets
             NOTE：（1): Due to the large data set, only a part of the data is left here for quick test code.
       
                    (2): The complete dataset can be downloaded from https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch.

        /graph_code_bert：Pre-trained models

        /java_fine_turn_GraBertlastepoch_lookLoss_GraBert：The backbone model after task fine-tuning

        /save_model：Save the trained model

        /src： source code
            /make_hard_samples: For making Hard Negative Samples
                /{lang}_adaptor_onceAllFile_2e4: Knowledge modules used for fusion

            /train_fusion: For training knowledge fusion modules
            /train_kms: For training knowledge modules
