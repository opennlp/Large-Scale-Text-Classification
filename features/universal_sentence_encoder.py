import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import numpy as np


def process_to_IDs_in_sparse_format(sp, sentences):
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return values, indices, dense_shape


def get_sentence_encoding(messages):
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
    encodings = module(
        inputs=dict(
            values=input_placeholder.values,
            indices=input_placeholder.indices,
            dense_shape=input_placeholder.dense_shape))

    with tf.Session() as sess:
      spm_path = sess.run(module(signature="spm_path"))

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(
          encodings,
          feed_dict={input_placeholder.values: values,
                    input_placeholder.indices: indices,
                    input_placeholder.dense_shape: dense_shape})

    return np.array(message_embeddings)

