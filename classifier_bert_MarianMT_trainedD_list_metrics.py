import torch
import numpy as np
import sklearn
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from discriminator_cnn_bert import Discriminator_cnn_bert  # Make sure to import your model class correctly
import dill
import logging
import os
import argparse
import options
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
from transformers import AutoTokenizer
from tqdm import tqdm


torch.cuda.empty_cache()

# CUDA multiple-GPU configuration

getpwd = os.getcwd()
# sys.path.append(
#     "/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master"
# )
sys.path.append(getpwd)
# https://stackoverflow.com/questions/67311527/how-to-set-gpu-count-to-0-using-os-environcuda-visible-devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.device_count() # result is 2

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.device_count() # result is 1, using first GPU

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.cuda.device_count() # result is 1, using second GPU"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#### Logging ####

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

parser = argparse.ArgumentParser(description="Adversarial-NMT-BERT")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)


"""
# # Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# def load_model(model_path, use_dill=False):
#     """
#     #Function to load the model.
#     #Set use_dill=True if the model was saved with dill.
#     """
#     if use_dill:
#         return torch.load(open(model_path, 'rb'), pickle_module=dill)
#     else:
#         # Assume args for Discriminator_cnn_bert initialization is provided or modify as needed
#         args = None  # Define this based on your model's requirements
#         model = Discriminator_cnn_bert(args)
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint['discriminator_state_dict'])
#         return model
def translate_with_max_length(text, max_length):
    import deep_translator
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="auto", target='fr')
    # Translate the text to French
    translated_text = translator.translate(text)

    # Check if the translated text is within the desired maximum length
    if len(translated_text) <= max_length:
        return translated_text
    else:
        # Truncate the translated text to the maximum length
        return translated_text[:max_length]

checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
import os
getpwd = os.getcwd()

model_tokenizer_configs = [ # /workspace/2024/git_repo_vastai/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G1_D_baseline_1_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G1_D_baseline_1_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G1_D_baseline_2_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G1_D_baseline_3_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G1_D3_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G1_D4_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G2_D_baseline_1_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G2_D_baseline_2_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G2_D_baseline_3_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    # {
    #     "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G10_D2_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    # },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G10_D3_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G10_D4_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
    {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G10_D6_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
   {
        "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G10_D7_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
    },
     {
       "checkpoints_path": getpwd + "/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb_G10_D9_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt"
   }

]


def preprocess_testData_MarianMT(data):
    max_length = 128
    # Initialize the BERT tokenizer
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    import pandas as pd
    en=list()
    fr=list()
    for element in data['translation']:
        # print("element: ", element)
        en.append(element['en'])
        fr.append(element['fr'] )
    
    json_data_df = pd.DataFrame(list(zip(en,fr)), columns=['src','target'])
    import deep_translator
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="auto", target='fr')
    
    import numpy as np
    import random
    from random import sample
    random.seed(12345)

    # given data frame df

    # create random index
    rindex =  np.array(sample(range(len(json_data_df)), int((len(json_data_df)/2))))
    for i, row in json_data_df.iterrows():
        if i in rindex:
            # Translate the 'src' column and limit to max_len 50
            translated_text = translate_with_max_length(row['src'], max_length)
            json_data_df.loc[i, 'ht_mt_target'] = translated_text
            json_data_df.loc[i, 'ht_mt_label'] = '0'
        else:
            # Use the original 'target' column
            json_data_df.loc[i, 'ht_mt_target'] = row['target']
            json_data_df.loc[i, 'ht_mt_label'] = '1'
        
    en = json_data_df['src'].tolist()
    fr = json_data_df['target'].tolist()
    ht_mt_target = json_data_df['ht_mt_target'].tolist()
    ht_mt_label = json_data_df['ht_mt_label'].tolist()
    
    # Tokenize the data
    inputs = tokenizer(
        en, truncation=True, padding="max_length", max_length=max_length
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            fr, truncation=True, padding="max_length", max_length=max_length
        )
        
        ht_mt_target = tokenizer(
            ht_mt_target, truncation=True, padding="max_length", max_length=max_length
        )

    
    #print statements for debugging
    print("inputs type: ", type(inputs))
    print("type of targets: ", type(targets))
    print("type of ht_mt_target: ", type(ht_mt_target))
    
    return {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "target_ids": targets['input_ids'],
        "target_attention_mask": targets['attention_mask'],
        "ht_mt_target_ids": ht_mt_target['input_ids'],
        "ht_mt_target_attention_mask": ht_mt_target['attention_mask'],
        "ht_mt_label": json_data_df['ht_mt_label'].tolist()  # Ensure this is a list
    } 
def preprocess_sm(data):
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    en=list()
    fr=list()
    for sentence_pair in data:
        # print("element: ", element)
        en.append(sentence_pair['en'])
        fr.append(sentence_pair['fr'] )
    
    # Tokenize the data
    inputs = tokenizer(
        en, truncation=True, padding="max_length", max_length=128
    )
    targets = tokenizer(
        fr, truncation=True, padding="max_length", max_length=128
    )

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


def predict_sm(model, text):
    """
    #Function to predict using the model.
    """
    model.eval()  # Set model to evaluation mode
    processed_text = preprocess_sm(text)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sample = sample.to(device)
    # Move all tensors in the batch to the device
    processed_text = {key: tensor.to(device) for key, tensor in processed_text.items()}

    # Get the source and target sentences from the batch
    src_sentences = processed_text["input_ids"]
    tgt_sentences = processed_text["target_ids"]

    with torch.no_grad():
        # Assuming your model's forward method can handle these inputs directly
        # You might need to adjust this based on your model's requirements
        output = model(src_sentences, tgt_sentences)

    return output


def predict(model, test_dataloader):
    """
    #Function to predict using the model.
    """
    model.eval()  # Set model to evaluation mode
    # processed_text = preprocess(text)
    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # sample = sample.to(device)
            # Move all tensors in the batch to the device
            sample = {key: tensor.to(device) for key, tensor in sample.items()}

            # Get the source and target sentences from the batch
            src_sentences = sample["input_ids"]
            tgt_sentences = sample["target_ids"]
            ht_mt_target = sample["ht_mt_target_ids"]
            ht_mt_label = sample["ht_mt_label"]
    
            # Assuming your model's forward method can handle these inputs directly
            # You might need to adjust this based on your model's requirements
            predicted_class = model(src_sentences, tgt_sentences)
            print(f"output: {predicted_class}")
            print("real class: ", ht_mt_label)
            
    return predicted_class


def main(args,checkpoints_path):
    
    use_cuda = torch.cuda.is_available()

    # Set model parameters
    args.encoder_embed_dim = 512 #768 #1000 # changed to 768 to match the BERT model
    args.encoder_layers = 2  # 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 512 #768 #1000 #changed to 768 to match the BERT model
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
    # Loading test dataset
    test_dataset = dataset["test"]  # Load the first 100 examples
    
    # Preprocess the data
    tokenized_test_datasets_translated = test_dataset.map(
        preprocess_testData_MarianMT, batched=True
    ) 

    print("tokenized_test_datasets_translated: ", tokenized_test_datasets_translated)
    
    # Path to your saved checkpoint
    # saved_model = 'validation_checkpoint_1.pt'
    # checkpoints_path = 'checkpoints/bert_dualG/wmt14_en_fr_20k/' + saved_model

    # checkpoints_path = '/home/paperspace/google_drive_v7/Research_Thesis/2024/Adversarial_NMT_th/checkpoints/best_discriminator_at_2.pt'
    # checkpoints_path = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/best_discriminator_at_2.pt'
    # checkpoints_path = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/best_discriminator_at_1.pt'
    # checkpoints_path = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/best_checkpoint_dict_format_2.pt'
    # checkpoints_path = '/home/paperspace/google_drive_v7/Research_Thesis/2024/Adversarial_NMT_th/checkpoints/bert_dualG/wmt14_en_fr_800sent_pg_kd_loss/best_discriminator.pt'
    # checkpoints_path = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1000sents_debug_Normalkd_comb2_save_open_direct_pretrained/best_discriminator_dill_direct_at_1.pt' # COMB-2

    # Load the saved state dictionaries
    # checkpoint_D = torch.load(checkpoints_path, pickle_module=dill)
    # print("checkpoint ", checkpoint)
    discriminator_cnn = Discriminator_cnn_bert(args, use_cuda=use_cuda)

    # checkpoint_D_dict = checkpoint_D.state_dict()

    # # Load the state dictionaries into the model
    # discriminator_cnn.load_state_dict(checkpoints_path['discriminator_state_dict'])
    # discriminator_cnn.load_state_dict(checkpoint.state_dict())

    print("checkpoints_path ", checkpoints_path)
   
    model_dict = discriminator_cnn.state_dict()
    model = torch.load(checkpoints_path)
    pretrained_dict = model.state_dict()
    print("pretrained_dict type: ", type(pretrained_dict))
    print("model : ",model)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    discriminator_cnn.load_state_dict(model_dict)
   
    

    discriminator_cnn = discriminator_cnn.module if hasattr(discriminator_cnn, "module") else discriminator_cnn

    
    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator_cnn = torch.nn.DataParallel(discriminator_cnn).cuda()
        else:
            discriminator_cnn.cuda()
    else:
        discriminator_cnn.cpu()

    # discriminator_cnn.eval()  # Set the model to evaluation mode

    # Example text to classify
    """
    test_data_sm = [{'en': 'Spectacular Wingsuit Jump Over Bogota',
  'fr': 'Spectaculaire saut en "wingsuit" au-dessus de Bogota'},
 {'en': 'Sportsman Jhonathan Florez jumped from a helicopter above Bogota, the capital of Colombia, on Thursday.',
  'fr': "Le sportif Jhonathan Florez a sauté jeudi d'un hélicoptère au-dessus de Bogota, la capitale colombienne."},
 {'en': 'Wearing a wingsuit, he flew past over the famous Monserrate Sanctuary at 160km/h. The sanctuary is located at an altitude of over 3000 meters and numerous spectators had gathered there to watch his exploit.',
  'fr': "Equipé d'un wingsuit (une combinaison munie d'ailes), il est passé à 160 km/h au-dessus du célèbre sanctuaire Monserrate, situé à plus de 3 000 mètres d'altitude, où de nombreux badauds s'étaient rassemblés pour observer son exploit."},
 {'en': 'A black box in your car?',
  'fr': 'Une boîte noire dans votre voiture\xa0?'},
 {'en': "As America's road planners struggle to find the cash to mend a crumbling highway system, many are beginning to see a solution in a little black box that fits neatly by the dashboard of your car.",
  'fr': "Alors que les planificateurs du réseau routier des États-Unis ont du mal à trouver l'argent nécessaire pour réparer l'infrastructure autoroutière en décrépitude, nombreux sont ceux qui entrevoient une solution sous forme d'une petite boîte noire qui se fixe au-dessus du tableau de bord de votre voiture."},
 {'en': "The devices, which track every mile a motorist drives and transmit that information to bureaucrats, are at the center of a controversial attempt in Washington and state planning offices to overhaul the outdated system for funding America's major roads.",
  'fr': "Les appareils, qui enregistrent tous les miles parcourus par un automobiliste et transmettent les informations aux fonctionnaires, sont au centre d'une tentative controversée à Washington et dans les bureaux gouvernementaux de la planification de remanier le système obsolète de financement des principales routes américaines."},
 {'en': 'The usually dull arena of highway planning has suddenly spawned intense debate and colorful alliances.',
  'fr': 'Le secteur généralement sans intérêt de la planification des grands axes a soudain provoqué un débat fort animé et des alliances mouvementées.'},
 {'en': 'Libertarians have joined environmental groups in lobbying to allow government to use the little boxes to keep track of the miles you drive, and possibly where you drive them - then use the information to draw up a tax bill.',
  'fr': 'Les libertaires ont rejoint des groupes écologistes pour faire pression afin que le gouvernement utilise les petites boîtes pour garder la trace des miles que vous parcourez, et éventuellement de la route sur laquelle vous circulez, puis utiliser les informations pour rédiger un projet de loi fiscal.'},
 {'en': 'The tea party is aghast.', 'fr': 'Le Tea Party est atterré.'}]
    
    """
    
    # Prediction
    # predictions = predict(discriminator_cnn, test_data_sm)
    
    # Converting the processed data to PyTorch Tensors
    tokenized_test_datasets_translated.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "target_ids",
            "target_attention_mask",
            "ht_mt_target_ids",
            "ht_mt_target_attention_mask",
            "ht_mt_label"
        ],
    )
        
    test_dataloader_translated = DataLoader(tokenized_test_datasets_translated, batch_size=1)
    print("test_dataloader_translated: ", test_dataloader_translated)
    # predictions = predict(discriminator_cnn, test_dataloader)
    discriminator_cnn.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, sample_translate in enumerate(test_dataloader_translated):
            
            print("sample before: ", sample_translate.keys())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # sample = sample.to(device)
            # Move all tensors in the batch to the device
            # sample_translate = {key: tensor.to(device) for key, tensor in sample_translate.items()}
            sample_translate = {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample_translate.items()}
            print("sample: ", sample_translate.keys())
            # Get the source and target sentences from the batch
            src_sentences = sample_translate["input_ids"]
            tgt_sentences = sample_translate["target_ids"]
            ht_mt_target = sample_translate["ht_mt_target_ids"] 
            ht_mt_label = sample_translate["ht_mt_label"]
    
            # Assuming your model's forward method can handle these inputs directly
            # You might need to adjust this based on your model's requirements
            disc_out = discriminator_cnn(src_sentences, ht_mt_target)
            # print("predicted_class squueze: ", predicted_class.squeeze().long().cpu().numpy())  # Convert to numpy array 
            print(f"output: {disc_out}")
            print("real class: ", ht_mt_label)
            probabilities = disc_out.squeeze().detach().cpu().numpy()
            predicted_labels = (np.atleast_1d(probabilities) > 0.5).astype(int)
            print("predicted_labels: ", predicted_labels)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
            y_pred.extend(predicted_labels)
            y_true.extend([int(label) for label in sample_translate["ht_mt_label"]])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)        

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1) # pos_label=1 is the label for the human translated text
    recall = recall_score(y_true, y_pred, pos_label=1) # pos_label=1 is the label for the human translated text
    conf_matrix = confusion_matrix(y_true, y_pred) 
    

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    return accuracy, precision, recall, conf_matrix
    
       
if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    # main(options)
    results = []
    for config in tqdm(model_tokenizer_configs,desc="Running models"):
        accuracy, precision, recall, conf_matrix = main(options, config["checkpoints_path"])
        results.append({
            "model": config["checkpoints_path"],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "conf_matrix": conf_matrix
        })
    
    # Optionally save results to a file or print them
    for result in results:
        print(result)

    with open("disc_checkpoints_results_G10_D5.txt", "w") as file:
        for result in results:
            file.write(f"{result}\n")
    