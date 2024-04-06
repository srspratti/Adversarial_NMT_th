import torch
import torch.nn.functional as F
from queue import PriorityQueue
from generator_tf_bert import TransformerModel_bert
from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.hidden = hiddenstate
        self.prevNode = previousNode
        self.wordId = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for reward
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward

def beam_search_single_sequence(model, src, src_mask, beam_width=10, topk=1):
    model.eval()
    with torch.no_grad():
        # Encoding source sequence
        encoder_out = model.encoder(model.src_bert_embed(src, attention_mask=src_mask).last_hidden_state)

        # Start with the start of the sentence token
        start_token_id = model.tgt_bert_embed.config.bos_token_id
        start_node = BeamSearchNode(hiddenstate=encoder_out, previousNode=None, wordId=start_token_id, logProb=0, length=1)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # Starting beam search
        nodes = PriorityQueue()
        nodes.put((-start_node.eval(), start_node))
        qsize = 1

        # Start beam search loop
        while True:
            # Give up when decoding takes too long
            if qsize > 2000: break

            # Fetch the best node
            score, n = nodes.get()
            decoder_input = torch.LongTensor([n.wordId]).to(src.device)

            if n.wordId == model.tgt_bert_embed.config.eos_token_id and n.prevNode != None:
                endnodes.append((score, n))
                # If we reached maximum number of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # Decode for one step using decoder
            decoder_output, _ = model(None, decoder_input.unsqueeze(0))
            log_prob, indexes = torch.topk(decoder_output, beam_width)

            # Put them into queue
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].item()
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(n.hidden, n, decoded_t, n.logp + log_p, n.length + 1)
                score = -node.eval()
                nodes.put((score, node))

            qsize += beam_width - 1

        # Choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=lambda x: x[0]):
            utterance = []
            utterance.append(n.wordId)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordId)

            utterance = utterance[::-1]
            utterances.append(utterance)

        return utterances
    
def beam_search_dataset(model, dataset, beam_width=10, topk=1):
    """
    Perform beam search over an entire dataset.

    :param model: The Transformer model instance
    :param dataset: The tokenized dataset
    :param beam_width: Beam width for the search
    :param topk: Number of top sequences to return for each input
    :return: A list of best sequences for the entire dataset
    """
    model.eval()
    results = []
    for data in dataset:
        src = torch.tensor(data['input_ids']).unsqueeze(0)  # Assuming batch_first=True, add batch dimension
        src_mask = torch.tensor(data['attention_mask']).unsqueeze(0) if 'attention_mask' in data else None

        # Move to the same device as model
        src = src.to(next(model.parameters()).device)
        if src_mask is not None:
            src_mask = src_mask.to(next(model.parameters()).device)
        
        translation = beam_search_single_sequence(model, src, src_mask, beam_width, topk)
        results.append(translation)

    return results

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


    # # Setup CUDA if available
    use_cuda = torch.cuda.is_available() and len(args.gpuid) >= 1
    # if use_cuda and args.gpuid:
    #     cuda.set_device(args.gpuid[0])

    
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


    # Load model checkpoint if specified
    if args.model_checkpoint_path:
        generator2_trained_saved.load_state_dict(torch.load(args.model_checkpoint_path))

    if use_cuda:
        generator2_trained_saved.cuda()
    else:
        generator2_trained_saved.cpu()

    # Generate translations
    # Note: You might need to adjust this part to match how your data is structured and how you want to handle batched generation
    # translations = translations = beam_search_dataset(generator2_trained_saved, src, src_mask, beam_width=5, topk=1)
    translations = beam_search_dataset(generator2_trained_saved, tokenized_test_datasets, beam_width=5, topk=1)
    print(translations)

    # Handle the output of the translations as needed
    # For example, write them to a file or print them out
    # This is placeholder logic and will need to be adapted based on your needs
    with open('output_translations.txt', 'w') as f:
        for translation in translations:
            # Process and format the translation output as needed
            f.write(str(translation) + '\n')

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning("Unknown arguments: {}".format(unknown))
    
    main(args)
