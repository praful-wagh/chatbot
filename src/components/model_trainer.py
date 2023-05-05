import os, sys
import pandas as pd, numpy as np
import re, string, unicodedata
from src.logger import log
from src.exception import CustomException
from src.utils import save_object, getPath
from dataclasses import dataclass
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts


contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "don't": "do not",
    "don’t": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y’all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "ain’t": "am not",
    "aren’t": "are not",
    "can’t": "cannot",
    "can’t’ve": "cannot have",
    "’cause": "because",
    "could’ve": "could have",
    "couldn’t": "could not",
    "couldn’t’ve": "could not have",
    "didn’t": "did not",
    "doesn’t": "does not",
    "don’t": "do not",
    "don’t": "do not",
    "hadn’t": "had not",
    "hadn’t’ve": "had not have",
    "hasn’t": "has not",
    "haven’t": "have not",
    "he’d": "he had",
    "he’d’ve": "he would have",
    "he’ll": "he will",
    "he’ll’ve": "he will have",
    "he’s": "he is",
    "how’d": "how did",
    "how’d’y": "how do you",
    "how’ll": "how will",
    "how’s": "how is",
    "i’d": "i would",
    "i’d’ve": "i would have",
    "i’ll": "i will",
    "i’ll’ve": "i will have",
    "i’m": "i am",
    "i’ve": "i have",
    "isn’t": "is not",
    "it’d": "it would",
    "it’d’ve": "it would have",
    "it’ll": "it will",
    "it’ll’ve": "it will have",
    "it’s": "it is",
    "let’s": "let us",
    "ma’am": "madam",
    "mayn’t": "may not",
    "might’ve": "might have",
    "mightn’t": "might not",
    "mightn’t’ve": "might not have",
    "must’ve": "must have",
    "mustn’t": "must not",
    "mustn’t’ve": "must not have",
    "needn’t": "need not",
    "needn’t’ve": "need not have",
    "o’clock": "of the clock",
    "oughtn’t": "ought not",
    "oughtn’t’ve": "ought not have",
    "shan’t": "shall not",
    "sha’n’t": "shall not",
    "shan’t’ve": "shall not have",
    "she’d": "she would",
    "she’d’ve": "she would have",
    "she’ll": "she will",
    "she’ll’ve": "she will have",
    "she’s": "she is",
    "should’ve": "should have",
    "shouldn’t": "should not",
    "shouldn’t’ve": "should not have",
    "so’ve": "so have",
    "so’s": "so is",
    "that’d": "that would",
    "that’d’ve": "that would have",
    "that’s": "that is",
    "there’d": "there would",
    "there’d’ve": "there would have",
    "there’s": "there is",
    "they’d": "they would",
    "they’d’ve": "they would have",
    "they’ll": "they will",
    "they’ll’ve": "they will have",
    "they’re": "they are",
    "they’ve": "they have",
    "to’ve": "to have",
    "wasn’t": "was not",
    "we’d": "we would",
    "we’d’ve": "we would have",
    "we’ll": "we will",
    "we’ll’ve": "we will have",
    "we’re": "we are",
    "we’ve": "we have",
    "weren’t": "were not",
    "what’ll": "what will",
    "what’ll’ve": "what will have",
    "what’re": "what are",
    "what’s": "what is",
    "what’ve": "what have",
    "when’s": "when is",
    "when’ve": "when have",
    "where’d": "where did",
    "where’s": "where is",
    "where’ve": "where have",
    "who’ll": "who will",
    "who’ll’ve": "who will have",
    "who’s": "who is",
    "who’ve": "who have",
    "why’s": "why is",
    "why’ve": "why have",
    "will’ve": "will have",
    "won’t": "will not",
    "won’t’ve": "will not have",
    "would’ve": "would have",
    "wouldn’t": "would not",
    "wouldn’t’ve": "would not have",
    "y’all": "you all",
    "y’all": "you all",
    "y’all’d": "you all would",
    "y’all’d’ve": "you all would have",
    "y’all’re": "you all are",
    "y’all’ve": "you all have",
    "you’d": "you would",
    "you’d’ve": "you would have",
    "you’ll": "you will",
    "you’ll’ve": "you will have",
    "you’re": "you are",
    "you’ve": "you have",
    "n't": "not",
    "n'": "ng",
    "'bout": "about",
    "'til": "until"
}
con_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

@dataclass
class ModelTrainerConfig:
    project_path = getPath()
    preprocessor_obj_file_path: str = os.path.join(project_path, 'artifacts', 'preprocessor.pkl')
    trained_model_file_path = os.path.join(project_path,'artifacts','model.pkl')

def unicode_to_ascii(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def expand_contractions(raw):
    def replace(match):
        return contractions_dict[match.group(0)]
    return con_re.sub(replace, raw)

def clean_text(text):
    text = text.lower()
    text = unicode_to_ascii(text)
    text = expand_contractions(text)
    text = ''.join(w for w in text if w not in string.punctuation).strip()
    text = "<sos> " + text + " <eos>"
    return text

def tokenize(lang):
    token = tf.keras.preprocessing.text.Tokenizer(filters='')
    token.fit_on_texts(lang)
    tensor = token.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, token

class Encoder(tf.keras.Model):
    def __init__(self, voc_size, emb_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(voc_size, emb_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        # out: A tensor of shape (batch_size, max_length, enc_units), which contains a vector of size enc_units for each word in the input sequence.
        # These vectors represent the hidden states of the GRU layer at each time step.
        out, state = self.gru(x, initial_state=hidden)
        return out, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, voc_size, emb_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(voc_size, emb_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(voc_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_lang.word_index['<sos>']] * batch_size, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(raw_data_path):
        try:
            df = pd.read_csv(r"C:\Users\praful.waghamode\PycharmProjects\chatbot\artifacts\data.csv")

            df.que = df.que.apply(clean_text)
            df.ans = df.ans.apply(clean_text)
            que = df.que.values.tolist()
            ans = df.ans.values.tolist()
            log('cleaned')
            global input_tensor, input_lang, target_tensor, target_lang
            input_tensor, input_lang = tokenize(que)
            target_tensor, target_lang = tokenize(ans)
            global max_len_target, max_len_inp
            max_len_target, max_len_inp = target_tensor.shape[1], input_tensor.shape[1]

            xtr, xte, ytr, yte = tts(input_tensor, target_tensor, test_size=.2, random_state=0)

            log('splitted')

            global encoder, decoder, batch_size, units
            buffer_size = len(xtr)
            batch_size = 64
            steps_per_epoch = buffer_size // batch_size
            emb_dim = 256
            units = 1024

            voc_in_size = len(input_lang.word_index) + 1
            voc_tar_size = len(target_lang.word_index) + 1

            ds = tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(buffer_size)
            ds = ds.batch(batch_size, drop_remainder=True)

            exm_in_bat, exm_tar_bat = next(iter(ds))

            log('encoder')
            encoder = Encoder(voc_in_size, emb_dim, units, batch_size)
            sam_hidden = encoder.initialize_hidden_state()
            sam_output, sam_hidden = encoder(exm_in_bat, sam_hidden)
            print('Encoder output shape: (batch size, sequence length, units) {}'.format(sam_output.shape))
            print('Encoder Hidden state shape: (batch size, units) {}'.format(sam_hidden.shape))

            log('attention layer')
            attention_layer = BahdanauAttention(10)
            attention_result, attention_weights = attention_layer(sam_hidden, sam_output)

            print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
            print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

            log('decoder')
            decoder = Decoder(voc_tar_size, emb_dim, units, batch_size)
            sample_decoder_output, _, _ = decoder(tf.random.uniform((batch_size, 1)), sam_hidden, sam_output)
            print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

            log('opt')
            global optimizer, loss_object
            optimizer = tf.keras.optimizers.Adam()
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

            log('epoch')
            epochs = 50
            for epoch in range(1, epochs + 1):
                enc_hidden = encoder.initialize_hidden_state()
                total_loss = 0
                for (batch, (inp, targ)) in enumerate(ds.take(steps_per_epoch)):
                    batch_loss = train_step(inp, targ, enc_hidden)
                    total_loss += batch_loss
                print('Epoch:', epoch, '  Loss:', total_loss / steps_per_epoch)


            log('input and target pkl files are saving...')
            save_object(os.path.join(getPath(), 'artifacts','input_.pkl'), input_lang)
            save_object(os.path.join(getPath(), 'artifacts','target_.pkl'), target_lang)

            log('models are saving...')
            tf.saved_model.save(encoder, os.path.join(getPath(), 'artifacts','encoder'))
            tf.saved_model.save(decoder, os.path.join(getPath(), 'artifacts','decoder'))
            log('models saved.')

        except Exception as e:
            raise CustomException(e, sys)

def remove_tags(sent):
    return sent.split('<start>')[-1].split('<end>')[0]