#!/usr/bin/env python
# coding: utf-8

import warnings
import torch
from torch import nn
warnings.filterwarnings(action='ignore')

from autocorrect import Correcter
from chatbot import trainBot, normalizeString, indexesFromSentence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
correct = Correcter()

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


# Hyperparameters
attn_model = 'dot'
load_checkpoint = 6000
max_length = 20
min_count = 3
model_name = "lionheart"
dropout = 0.1
batch_size = 64
clip = 50
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5
n_iteration = 2000
print_every = 100
save_every = 2000


encoder, decoder, voc = trainBot(attn_model, load_checkpoint, max_length, min_count, model_name, dropout, 
                                    batch_size, clip, teacher_forcing_ratio, 
                                    learning_rate, decoder_learning_ratio, n_iteration, print_every, save_every)


# In[ ]:


def evaluate(encoder, decoder, searcher, voc, sentence, max_length):
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': 
                print("Goodbye!")
                break

            # TODO: Increment autocorrect file count
            output = []
            correct_input_sentence = correct(input_sentence.lower())
            
            if correct_input_sentence != input_sentence.lower():
                input_sentence = correct_input_sentence
                
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, max_length)
            
            # Format and print response sentence
            for word in output_words:
                if word == "i":
                    word = word.capitalize()
                if word != 'EOS':
                    output.append(word)
                else:
                    break
            
            output[0] = output[0].capitalize()
            if output[-1] == "." or output[-1] == "!" or output[-1] == "?":
                print('Bot:', ' '.join(output[:-1]) + output[-1])
            else:
                print('Bot:', ' '.join(output) + '.')

        except KeyError as key:
            key = str(key).strip('\'') # Strip the starting and ending quotation mark
            print(f"Bot: {key}?")


# In[ ]:


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# In[ ]:


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)


# In[ ]:





# In[ ]:




