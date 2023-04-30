import os.path
import sys
from src.exception import CustomException
from src.utils import load_object, getPath
from src.logger import log
from src.components import model_trainer
import tensorflow as tf

class PredictPipeline:

    def load_model(self):
       try:
           log('Loading input_ and target_')
           global input_, target_, encoder_, decoder_
           input_ = load_object(os.path.join(getPath(), 'artifacts','input_.pkl'))
           target_ = load_object(os.path.join(getPath(), 'artifacts','target_.pkl'))

           log('Loading encoder and decoder')
           encoder_ = tf.saved_model.load(os.path.join(getPath(), 'artifacts','encoder'))
           decoder_ = tf.saved_model.load(os.path.join(getPath(), 'artifacts','decoder'))

           log('Loading completed..')

       except Exception as e:
           raise CustomException(e,sys)


    def ask(self, msg):
        try:
            msg = model_trainer.clean_text(msg)
            inputs = [input_.word_index[i] for i in msg.split(' ')]
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=22, padding='post')
            inputs = tf.convert_to_tensor(inputs)
            result = ''
            # hidden = [tf.zeros((1,1024))]

            x_spec = tf.TensorSpec(shape=(None, 22), dtype=tf.int32, name='x')
            # hidden_spec = tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name='hidden')

            x = tf.ensure_shape(inputs, x_spec.shape)
            # hidden = tf.ensure_shape(dec_hidden, hidden_spec.shape)

            # x = tf.ones(shape=(1, 22), dtype=tf.int32)
            hidden = tf.zeros(shape=(1, 1024), dtype=tf.float32)

            enc_out, enc_hidden = encoder_(x, hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([target_.word_index['<sos>']],0)

            x_spec = tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name='x')
            hidden_spec = tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name='hidden')
            enc_output_spec = tf.TensorSpec(shape=(None, 22, 1024), dtype=tf.float32, name='enc_output')

            x_dec = tf.constant(dec_input, dtype=tf.int32)
            x_dec = tf.cast(x_dec, tf.float32)

            dec_input = tf.ensure_shape(x_dec, x_spec.shape)
            hidden = tf.ensure_shape(dec_hidden, hidden_spec.shape)
            enc_output = tf.ensure_shape(enc_out, enc_output_spec.shape)

            for t in range(22):
                predictions, dec_hidden, attention_weights = decoder_(dec_input, hidden, enc_output)
                attention_weights = tf.reshape(attention_weights, (-1, ))
                predicted_id = tf.argmax(predictions[0]).numpy()
                result += target_.index_word[predicted_id] + ' '
                if target_.index_word[predicted_id] == '<eos>':
                    return model_trainer.remove_tags(result)
                dec_input = tf.expand_dims([predicted_id],0)
                dec_input = tf.constant(dec_input, dtype=tf.int32)
                dec_input = tf.cast(dec_input, tf.float32)

        except Exception as e:
            CustomException(e, sys)