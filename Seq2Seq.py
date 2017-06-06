import tensofrlow as tf
import numpy as np
from distutils.version import LooseVersion
import warnings

assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def model_inputs():
    input_ = tf.placeholder(tf.int32, [None, None], 'input')
    target_ = tf.placeholder(tf.int32, [None, None], 'target')
    learning_rate_ = tf.placeholder(tf.float32, None, 'lr')
    keep_prob_ = tf.placeholder(tf.float32, None, 'keep_prob')
    return input_, target_, learning_rate_, keep_prob_
    
    
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
    outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
    return final_state
 
 
 
 def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    simple_dec_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)    
    outputs_train, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(cell = dec_cell, 
                                                                 decoder_fn = simple_dec_fn_train, 
                                                                 inputs = dec_embed_input, 
                                                                 sequence_length = sequence_length, 
                                                                 scope=decoding_scope)                                                                                     
    logits_train = output_fn(outputs_train)
    logits = tf.nn.dropout(logits_train, keep_prob)
    
    return logits
    
 def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    simple_dec_fn_infer = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn,
                                                                  encoder_state,
                                                                  dec_embeddings,
                                                                  start_of_sequence_id, 
                                                                  end_of_sequence_id,
                                                                  maximum_length,
                                                                  vocab_size)
    dropout = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
  
    logits_infer,final_state, final_context_state = \
                         tf.contrib.seq2seq.dynamic_rnn_decoder(dropout,
                                                                simple_dec_fn_infer,
                                                                sequence_length=maximum_length,
                                                                scope=decoding_scope) 
    return logits_infer   
    



def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    dec_cell = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
    max_target_sentence_length = max([len(sentence) for sentence in source_int_text])
    
    with tf.variable_scope('decoding_layer') as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, num_outputs=vocab_size,
                                                                activation_fn = None,
                                                                scope=decoding_scope)
    
    logits_train = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                                            sequence_length, decoding_scope, output_fn, keep_prob)
       

    decoding_scope.reuse_variables()
    logits_infer = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, 
                                            target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], 
                                            max_target_sentence_length, 
                                            vocab_size, decoding_scope, output_fn, keep_prob)
         
    return logits_train, logits_infer
    
    
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    This function builds the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,\
                                                       source_vocab_size,\
                                                       enc_embedding_size)
   
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.truncated_normal([target_vocab_size, dec_embedding_size], stddev=0.01))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    logits_train, logits_infer = decoding_layer(dec_embed_input, dec_embeddings, 
                                                enc_state, target_vocab_size, 
                                                sequence_length, rnn_size, num_layers, 
                                                target_vocab_to_int, keep_prob)
    
    return logits_train, logits_infer

    

