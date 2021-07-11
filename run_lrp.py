import imp

import numpy as np
import pandas as pd
import os
import re
import pickle
import time
import string
import json
import random
import argparse

import keras
import keras.backend
import keras.models

from keras.preprocessing import text, sequence
from keras.datasets import mnist
from keras.models import Model
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from gensim.models.doc2vec import Doc2Vec

import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from matplotlib.font_manager import FontProperties

import innvestigate
import innvestigate.applications
import innvestigate.applications.mnist
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from innvestigate.utils.tests.networks import base as network_base

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def text_pipeline(x):
    # remove embedding ad script
    x = re.sub(r'"use strict";.*\n','',x)
    x = re.sub("\/{2}\s<!\[CDATA\[\n.*\n\/{2}\s\]{2}>", '', x)
    # remove hypyerlinks
    x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)
    # remove ignore character
    x = re.sub('[\n│-]','',x)
    x = re.sub('\xa0', '', x)
    x = re.sub('\u3000', '', x)
    x = re.sub('\u200b', '', x)
    # remove special character
    x = re.sub('→↓△▿⋄•！？?〞＃＄％＆』（）＊＋，－╱︰；＜＝＞＠〔╲〕 ＿ˋ｛∣｝∼、〃》「」『』【】﹝﹞【】〝〞–—『』「」…﹏', '',x)
    x = x.translate(str.maketrans('', '', string.punctuation))
    return x

def remove_slogan(x):
    """
    Namelist which is no slogan: jackla39,jesse0218,mei30530,as660707,saliha,thudadai,brainfart99
    """
    # z78625
    x = re.sub('如果您喜歡這篇文章，請幫海綿飽飽按讚鼓勵，感謝您歡迎加入海綿飽飽的facebook粉絲團海綿飽飽的鳳梨城堡', '', x)
    # masaharuwu
    x = re.sub("那個...不才小菲也害羞的申請一個粉絲團了~   再請大家幫小菲點進去按個讚唷!!感恩感恩!", '', x)
    # may1215may 
    x = re.sub("歡迎加入小妞的生活旅程粉絲團，最新消息不漏接喔!", '', x)
    # jeremyckt2
    x = re.sub("快速追蹤「Jeremy以食為天」！ 大家在IG上也一起來追蹤「Jeremy以食為天」吧！帳號： jeremyfoodie歡迎來follow我的美食粉絲頁喔Jeremy以食為天在愛食記，歡迎大家追蹤！", '', x)
    # shinylu0920    181
    x = re.sub("FB搜尋 :我是呂萱萱(點我)", '', x)
    # eggface45
    x = re.sub("@蛋寶趴趴go", '', x)
    # reinmiso
    x = re.sub("想看更多分享嗎?別忘了點讚FB / 訂閱Blog / 追蹤IG 唷！", '', x)
    return x

def make_train_val_test(df):
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    test_idx = df[~msk].index

    val_msk = np.random.rand(len(train_df)) < 0.8
    train_idx = train_df[val_msk].index
    val_idx = train_df[~val_msk].index
    
    print("Train: {} / Validation: {} / Test: {}".format(len(train_idx), len(val_idx), len(test_idx)))
    return train_idx, val_idx, test_idx
    
def prepare_dataset(ds):
    filtered_indices = df.splitset_label == SPLIT_LABEL_MAPPING[ds]
    
    reviews_in_ds = df[filtered_indices]
    
    xd = np.zeros((len(reviews_in_ds), MAX_SEQ_LENGTH, EMBEDDING_DIM))
    y = reviews_in_ds.label_idx.values.astype(int)
    # over_y = reviews_in_ds.y.values.astype(int)

    reviews = []
    for i, seg in enumerate(reviews_in_ds.seg.values):
        # sostr = sostr.lower()
        review = []
        # for j, v in enumerate(seg.split(' ')[-MAX_SEQ_LENGTH:]): # post-cut
        for j, v in enumerate(seg.split(' ')[:MAX_SEQ_LENGTH]): # pre-cut
            if v in doc2vec.wv.vocab:
                e_idx = encoder[v]
                xd[i, j, :] = doc2vec.wv[v]
            else:
                e_idx = 0
            
            review.append(e_idx)
        reviews.append(review)
        

    return dict(
        x4d=np.expand_dims(xd, axis=1),
        y=y,
        encoded_reviews=reviews,
    )
    

def build_network(input_shape, output_n, activation=None, dense_unit=256, dropout_rate=0.25):
    if activation:
        activation = "relu"

    net = {}
    net["in"] = network_base.input_layer(shape=input_shape)
    # net["emb"] = tf.keras.layers.Embedding(max_features, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(net["in"])
    net["conv"] = keras.layers.Conv2D(filters=100,kernel_size=(1,3), strides=(1, 1), padding='valid')(net["in"])
    net["pool"] = keras.layers.MaxPooling2D(pool_size=(1, input_shape[2]-2))(net["conv"])
    net["out"] = network_base.dense_layer(keras.layers.Flatten()(net["pool"]), units=output_n, activation=activation)
    net["sm_out"] = network_base.softmax(net["out"])


    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def to_one_hot(y):
    return keras.utils.to_categorical(y, NUM_CLASSES)

def train_model(model,  batch_size=128, epochs=20):
    
    x_train = DATASETS['training']['x4d']
    y_train = to_one_hot(DATASETS['training']['y'])

    x_test = DATASETS['testing']['x4d']
    y_test = to_one_hot(DATASETS['testing']['y'])
    
    x_val = DATASETS['validation']['x4d']
    y_val = to_one_hot(DATASETS['validation']['y'])
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        shuffle=True
                       )
    score = model.evaluate(x_test, y_test, verbose=0)
    # score = model.evaluate(x_val_, y_val_, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    plt.show()

def plot_text_heatmap(words, scores, title="", width=10, height=0.2, verbose=0, max_word_per_line=20):
    """
    This is a utility method visualizing the relevance scores of each word to the network's prediction. 
    one might skip understanding the function, and see its output first.
    """
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    ax.set_title(title, loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)
    
    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t, fontproperties=myfont)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+15, units='dots')

    if verbose == 0:
        ax.axis('off')
        
def top_words_in_lrp(test, decoder, k):
    top_words_pos = {}
    top_words_neg = {}
    
    for i, idx in enumerate(test):

        words = [decoder[t] for t in list(DATASETS['testing']['encoded_reviews'][idx])]

        # print('Review(id=%d): %s' % (idx, ' '.join(words)))
        y_true = DATASETS['testing']['y'][idx]
        y_pred = test_sample_preds[i]

        # print("Pred class : %s %s" %
        #       (LABEL_IDX_TO_NAME[y_pred], '✓' if y_pred == y_true else '✗ (%s)' % LABEL_IDX_TO_NAME[y_true])
        #      )

        for j in list(analysis[i, 2].reshape(-1).argsort()[-10:][::-1]):
            try:
                top_words_pos[words[j]] = top_words_pos.get(words[j], 0) + 1
            except:
                # print("cannot find word index", j)
                pass

        for j in list(analysis[i, 2].reshape(-1).argsort()[:10][::-1]):
            try:
                top_words_neg[words[j]] = top_words_neg.get(words[j], 0) + 1
            except:
                # print("cannot find word index", j)
                pass
        # for j, method in enumerate(methods):
        #     plot_text_heatmap(words, analysis[i, j].reshape(-1), title='Method: %s' % method, verbose=0)
        #     plt.show()

    top_words_pos_k = sorted(top_words_pos.items(), key=lambda x: x[1], reverse=True)[:k]
    top_words_neg_k = sorted(top_words_neg.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return top_words_pos_k, top_words_neg_k

def plot_pos_neg_text(top_words_pos_10, top_words_neg_10):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    fig.subplots_adjust(wspace=0.8)

    ax1.barh(range(len(top_words_neg_10)), [w[1] for w in top_words_neg_10], align='center')
    ax1.set_title('Negative contribution', fontsize=18)
    plt.sca(ax1)
    plt.yticks(range(len(top_words_neg_10)), [w[0] for w in top_words_neg_10], fontsize=16)
    ax1.yaxis.tick_right()
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()


    ax2.barh(range(len(top_words_pos_10)), [w[1] for w in top_words_pos_10], align='center', color='red')
    ax2.set_title('Positive contribution', fontsize=18)
    plt.sca(ax2)
    plt.yticks(range(len(top_words_pos_10)), [w[0] for w in top_words_pos_10], fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run CNN LRP.')    
    parser.add_argument('--data_path', type=str, help='path of original data')
    parser.add_argument('--embedding_path', type=str, help='path of word embedding, only supportive of doc2vec currenlty')
    parser.add_argument('--pos_neg_flag', default='positive', help="print out whether positive or negative result")
    args = parser.parse_args()

    data_path = args.data_path
    embedding_path = args.embedding_path
    pos_neg_flag = args.pos_neg_flag
    MAX_SEQ_LENGTH = 300
    EMBEDDING_DIM = 300
    LABEL_MAPPING = {"業配":1, "Unknown":0}
    NUM_CLASSES = len(set(LABEL_MAPPING.keys()))
    
    with open(data_path, "r") as f:
        train = json.load(f)

    content = [t['content'] for t in train]
    seg = [text_pipeline(t['word_seg']) for t in train]
    label = [t['label'] for t in train]

    df = pd.DataFrame([{'content': c, 'seg': s, 'label': l} for c, s, l in zip(content, seg, label)]).reset_index(drop=True)
    df['label_idx'] = df.label.map(LABEL_MAPPING)

    train_idx, val_idx, test_idx = make_train_val_test(df)

    df['splitset_label'] = 1
    df.iloc[test_idx, 4] = 2
    df.iloc[val_idx, 4] = 3
    
    doc2vec = Doc2Vec.load(embedding_path)
    # Unknown vocabs are set to <UNK>.
    encoder = dict(zip(['<UNK>'] + doc2vec.wv.index2word, range(0, len(doc2vec.wv.index2word) +1)))
    # encoder = dict(zip(doc2vec.wv.index2word, range(0, len(doc2vec.wv.index2word))))
    decoder = dict(zip(encoder.values(), encoder.keys()))
    SPLIT_LABEL_MAPPING = {
        'training' : 1,
        'testing': 2,
        'validation': 3
    }
    DATASETS = dict()

    for ds in ['training', 'testing', 'validation']:
        DATASETS[ds] = prepare_dataset(ds)

    print('We have {} reviews in the training set, and {} reviews in the testing set'.format
          (len(DATASETS['training']['x4d']), len(DATASETS['testing']['x4d'])))

    net = build_network((None, 1, MAX_SEQ_LENGTH, EMBEDDING_DIM), NUM_CLASSES)
    model_without_softmax, model_with_softmax = Model(inputs=net['in'], outputs=net['out']), Model(inputs=net['in'], outputs=net['sm_out'])

    train_model(model_with_softmax, batch_size=128, epochs=10)
    model_without_softmax.set_weights(model_with_softmax.get_weights())

    # Specify methods that you would like to use to explain the model. 
    # Please refer to iNNvestigate's documents for available methods.
    methods = ['gradient', 'lrp.z', 'lrp.alpha_2_beta_1', 'pattern.attribution']
    kwargs = [{}, {}, {}, {'pattern_type': 'relu'}]
    # build an analyzer for each method
    analyzers = []

    for method, kws in zip(methods, kwargs):
        analyzer = innvestigate.create_analyzer(method, model_without_softmax, **kws)
        analyzer.fit(DATASETS['training']['x4d'], batch_size=256, verbose=1)
        analyzers.append(analyzer)

    test_all_preds = [None] * len(test_idx)
    for i in range(len(test_idx)):
        x, y = DATASETS['testing']['x4d'][i], DATASETS['testing']['y'][i]
        x = x.reshape((1, 1, MAX_SEQ_LENGTH, EMBEDDING_DIM))    
        prob = model_with_softmax.predict_on_batch(x)[0] #forward pass with softmax
        y_hat = prob.argmax()
        test_all_preds[i] = y_hat


    tp, fp, fn, tn = [], [], [], []
    for j in range(len(test_idx)):
        if test_all_preds[j] == 1 and DATASETS['testing']['y'][j] == 0:
            fn.append(j)
        if test_all_preds[j] == 0 and DATASETS['testing']['y'][j] == 1:
            fp.append(j)
        if test_all_preds[j] == 1 and DATASETS['testing']['y'][j] == 1:
            tp.append(j)
        if test_all_preds[j] == 0 and DATASETS['testing']['y'][j] == 0:
            tn.append(j)

    print(classification_report(DATASETS['testing']['y'], test_all_preds, digits=5))

    matrix = confusion_matrix(DATASETS['testing']['y'],test_all_preds)
    plot_confusion_matrix(matrix, ['Unknown', 'Sponsorship'])

    LABEL_IDX_TO_NAME = {
        0: 'Unknown',
        1: 'Sponsorship'
    }

    if pos_neg_flag == "positive":
        test = tp + fp
    else:
        test = tn + fn

    test_sample_preds = [None]*len(test)
    analysis = np.zeros([len(test), len(analyzers), 1, MAX_SEQ_LENGTH])

    for i, ridx in enumerate(test):

        x, y = DATASETS['testing']['x4d'][ridx], DATASETS['testing']['y'][ridx]

        t_start = time.time()
        x = x.reshape((1, 1, MAX_SEQ_LENGTH, EMBEDDING_DIM))    

        presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
        prob = model_with_softmax.predict_on_batch(x)[0] #forward pass with softmax
        y_hat = prob.argmax()
        test_sample_preds[i] = y_hat

        for aidx, analyzer in enumerate(analyzers):

            a = np.squeeze(analyzer.analyze(x))
            a = np.sum(a, axis=1)

            analysis[i, aidx] = a
        t_elapsed = time.time() - t_start
        # print('Review %d (%.4fs)'% (ridx, t_elapsed))

    top_words_pos_20, top_words_neg_20 = top_words_in_lrp(test, decoder, 20)
    plot_pos_neg_text(top_words_pos_20, top_words_neg_20)
