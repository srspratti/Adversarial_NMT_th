import argparse
import logging
import os
import torch
from torch import cuda
from transformers import BertTokenizerFast, BertModel
import options
import data
from torch.utils.data import DataLoader
from sequence_generator import SequenceGenerator

from datasets import load_dataset
from generator_tf_bert import TransformerModel_bert
# from preprocess import preprocess

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Sequence generation with TransformerModel_bert.")

# Add arguments as needed, following the structure of your original script
# This is just a placeholder structure for the arguments
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_checkpoint_args(parser)
options.add_distributed_training_args(parser)
options.add_generation_args(parser)
options.add_generator_model_args(parser)


def main(args):

    # Set model parameters
    args.encoder_embed_dim = 768  # 1000 # changed to 768 to match the BERT model
    args.encoder_layers = 2  # 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 768  # 1000 #changed to 768 to match the BERT model
    args.encoder_heads = 2
    args.encoder_ffn_embed_dim = 1000

    args.decoder_heads = 2
    args.decoder_ffn_embed_dim = 1000
    args.decoder_layers = 2  # 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    # Loading data using datasets library from the HuggingFace
    # Get user's home directory
    import os
    home = os.path.expanduser("~")

    # Define the path of the cache directory
    cache_dir = os.path.join(home, ".cache", "huggingface", "datasets")

    # Define the name and configuration of the dataset
    dataset_name = "wmt14"
    config_name = "fr-en"

    # Build the path for the specific dataset configuration
    dataset_config_path = os.path.join(cache_dir, dataset_name, config_name)

    print(f"Checking cache at: {dataset_config_path}")

    # Check if the dataset configuration is already cached
    if os.path.exists(dataset_config_path) and len(os.listdir(dataset_config_path)) > 0:
        print("Dataset already downloaded, loading from cache.")
        # If the dataset is already downloaded, load it from the cache directory
        dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
    else:
        print("Downloading the dataset.")
        # Download the dataset and specify the cache directory
        dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)

    # Here, you should adjust the loading of subsets to avoid redundant downloads or loading.
    # Load 50k rows of the train dataset
    train_dataset = dataset["train"].select(range(100020))
    # train_dataset = dataset["train"].select(range(600))

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Loading Bert Model
    bert_model = "bert-base-multilingual-cased"

    # Pre-processing the data
    # To-Do : Need to change the max_length to 50 from 128
    def preprocess(data):
        # Initialize the BERT tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

        en = list()
        fr = list()
        for element in data["translation"]:
            # print("element: ", element)
            en.append(element["en"])
            fr.append(element["fr"])

        # Tokenize the data
        inputs = tokenizer(en, truncation=True, padding="max_length", max_length=128)
        targets = tokenizer(fr, truncation=True, padding="max_length", max_length=128)

        # Convert tokens to their corresponding IDs
        input_ids = inputs.input_ids
        target_ids = targets.input_ids

        # Create attention masks
        input_attention_mask = inputs.attention_mask
        target_attention_mask = targets.attention_mask

        return { 
            "input_ids": input_ids,
            "attention_mask": input_attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
        }

    # print(train_dataset[0])
    # tokenized_datasets = dataset.map(tokenize_function, batched=True) # using the other berttokenizer map function
    tokenized_train_datasets = train_dataset.map(
        preprocess, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function
    tokenized_valid_datasets = valid_dataset.map(
        preprocess, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function
    tokenized_test_datasets = test_dataset.map(
        preprocess, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function


    # Setup CUDA if available
    use_cuda = torch.cuda.is_available() 
    if use_cuda and args.gpuid:
        cuda.set_device(args.gpuid[0])

     # loeading the pre-trained model 

    # generator2_trained_saved.load_state_dict(torch.load(args.model_checkpoint_path))
    g_model_path = '/home/paperspace/google_drive_v4/Research_Thesis/2024/Adversarial_NMT_th/checkpoints/bert_dualG/wmt14_en_fr_10sent/best_generator.pt'
    assert os.path.exists(g_model_path)
    generator2_trained_saved = TransformerModel_bert(args, use_cuda=use_cuda)  
    model_dict = generator2_trained_saved.state_dict()
    model = torch.load(g_model_path)
    pretrained_dict = model.state_dict()
    print("pretrained_dict type: ", type(pretrained_dict))
    print("model : ",model)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator2_trained_saved.load_state_dict(model_dict)

    # Initialize the model
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    # bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    if use_cuda:
        if torch.cuda.device_count() > 1:
            generator2_trained_saved = torch.nn.DataParallel(generator2_trained_saved).cuda()
        else:
            generator2_trained_saved.cuda()
    else:
        generator2_trained_saved.cpu()


    batch_size = args.joint_batch_size

    # Create a DataLoader for the train data
    test_dataloader = DataLoader(tokenized_test_datasets, batch_size=batch_size)

    # Initialize the sequence generator
    translator = SequenceGenerator(
        model=generator2_trained_saved,
        beam_size=args.beam,
        stop_early=not args.no_early_stop,
        normalize_scores=not args.unnormalized,
        len_penalty=args.lenpen,
        unk_penalty=args.unkpen
    )

    if use_cuda:
        translator.cuda()


    # Generate translations
    # Note: You might need to adjust this part to match how your data is structured and how you want to handle batched generation
    translations = translator.generate_batched_itr(
        test_dataloader,
        maxlen_a=args.max_len_a,
        maxlen_b=args.max_len_b,
        cuda=use_cuda
    )


    # Handle the output of the translations as needed
    # For example, write them to a file or print them out
    # This is placeholder logic and will need to be adapted based on your needs
    # with open('output_translations.txt', 'w') as f:
    #     for translation in translations:
    #         # Process and format the translation output as needed
    #         f.write(str(translation) + '\n')

    with open('predictions_fr_en.txt', 'wb') as translation_writer:
        with open('real_fr_en.txt', 'wb') as ground_truth_writer:

            translations = translator.generate_batched_itr(
                test_dataloader, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b, cuda=use_cuda)

            for sample_id, src_tokens, target_tokens, hypos in translations:
                # Process input and ground truth
                target_tokens = target_tokens.int().cpu()
                src_str = dataset.src_dict.string(src_tokens, args.remove_bpe)
                target_str = dataset.dst_dict.string(
                    target_tokens, args.remove_bpe, escape_unk=True)

                # Process top predictions
                for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                    hypo_tokens = hypo['tokens'].int().cpu()
                    hypo_str = dataset.dst_dict.string(
                        hypo_tokens, args.remove_bpe)

                    hypo_str += '\n'
                    target_str += '\n'

                    translation_writer.write(hypo_str.encode('utf-8'))
                    ground_truth_writer.write(target_str.encode('utf-8'))

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning("Unknown arguments: {}".format(unknown))
    main(args)