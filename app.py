from flask import Flask, Response
from flask import render_template, jsonify, request
import io
import re
import base64
import numpy as np
import pickle
import keras
from keras import backend as K
from keras.models import Model
from gensim.models.doc2vec import Doc2Vec
import innvestigate
from innvestigate.utils.tests.networks import base as network_base
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
import tensorflow as tf
import warnings
import logging
import jieba
warnings.filterwarnings("ignore")


app = Flask(__name__)
handler = logging.FileHandler("test.log")  # Create the file logger
app.logger.addHandler(handler)             # Add it to the built-in logger
app.logger.setLevel(logging.DEBUG)         # Set the log level to debug
stopwords = []
with open("dict/cn_stopwords.txt", "r")  as f:
    for line in f.readlines():
        stopwords.append(line)

MAX_SEQ_LENGTH = 300
EMBEDDING_DIM = 300

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

test_content = "推薦到 捷運 中山站 一帶 逛逛 朋友 們 順道 站 內 七盞茶 買 杯 飲料 品嘗 看看 此 篇 為 合作文 並且 為 真實 用餐 感受"

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

def text_cleasing(x):
    x = re.sub(r'"use strict";.*\n','',x)
    x = re.sub("\/{2}\s<!\[CDATA\[\n.*\n\/{2}\s\]{2}>", '', x)
                    
    x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)
    x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)
    x = re.sub('[\n│-]','',x)
    x = re.sub('\xa0', '', x)
    x = re.sub('\u3000', '', x)
    x = re.sub('\u200b', '', x)
    return x

def tokenizer(content):
    xd = np.zeros((1, MAX_SEQ_LENGTH, EMBEDDING_DIM))
    # change
    vectors = []
    for j,v in enumerate(content[-MAX_SEQ_LENGTH:]):
        if v in doc2vec.wv.vocab:
            e_idx = encoder[v]
            xd[0,j,:] = doc2vec.wv[v]
        else:
            e_idx = 0
        vectors.append(e_idx)

    return np.expand_dims(xd, axis=1)

def plot_text_heatmap(words, scores, title="", width=10, height=2.5, verbose=0, max_word_per_line=20):

    """
    This is a utility method visualizing the relevance scores of each word to the network's prediction. 
    one might skip understanding the function, and see its output first.
    """
    fig, ax = plt.subplots(1,1,figsize=(width, height))
    
    #ax = plt.gca()

    #ax.set_title(title, loc='left')
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
    # loc_y = -0.2
    loc_y = 0.0

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        # create a new line if thll line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y - 2.5
            t = ax.transData
            ax.set_ybound(loc_y)
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+15, units='dots')

    if verbose == 0:
        ax.axis('off')

    return fig

print("loading do2vec")
global doc2vec
doc2vec = Doc2Vec.load("embeddings/doc2vec.bin")
encoder = dict(zip(['<UNK>'] + doc2vec.wv.index2word, range(0, len(doc2vec.wv.index2word) + 1)))
decoder = dict(zip(encoder.values(), encoder.keys()))

print("load keras model")
net = build_network((None, 1, MAX_SEQ_LENGTH, EMBEDDING_DIM), 2)
model_with_softmax = keras.models.load_model("models/cnn_with_softmax.h5")
model_without_softmax = Model(inputs=net['in'], outputs=net['out'])
model_without_softmax.set_weights(model_with_softmax.get_weights())

print("load analyzer")
global analyzer
analyzer = innvestigate.create_analyzer("lrp.alpha_2_beta_1", model_without_softmax)
graph = tf.get_default_graph()

def predict_and_lrp(input_content):
    x = tokenizer(input_content)
    x = x.reshape((1, 1, MAX_SEQ_LENGTH, EMBEDDING_DIM)) 
    with graph.as_default():
        #presm = model_without_softmax.predict_on_batch(x)[0]
        prob = model_with_softmax.predict_on_batch(x)[0]
        a = np.squeeze(analyzer.analyze(x))

    y_hat = prob.argmax()
    y_prob = prob[y_hat]
    #app.logger.info(y_hat)
    a = np.squeeze(analyzer.analyze(x))
    #app.logger.info(a)
    a = np.sum(a, axis=1)

    return y_hat, y_prob, a


@app.route("/")
def hello_world():
    buf = io.BytesIO()
    # fig.savefig(buf, format="png")
    #data = base64.b64encode(buf.getbuffer()).decode("ascii")
    #return f"<img src='data:image/png;base64,{data}'/>"
    return render_template("index.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        search = request.get_json()
        #fig = plot_text_heatmap(words, a.reshape(-1), title="lrp alpha beta  method", verbose=0)
        #buf = io.BytesIO()
        #fig.savefig(buf, format="png")
        #img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        #search['img_data']=img_data
        return jsonify(search)
    return render_template("index.html")

@app.route("/lrp_plot", methods=['GET','POST'])
def lrp_plot():
    if request.method == 'POST':
        received_data = request.get_json()
        # y_hat, a = predict_and_lrp(received_data['text'])
        #words = received_data["text"].split(' ')
        clean_content = text_cleasing(str(received_data['text']))
        words = [w for w in jieba.cut(clean_content) if w not in stopwords]
        y_hat, y_prob, a = predict_and_lrp(words)
        words = words[-MAX_SEQ_LENGTH:]
        fig = plot_text_heatmap(words, a.reshape(-1), title="lrp alpha beta  method", verbose=0)
        #fig.savefig("/static/img/lrp_plot.png")
        buf = io.BytesIO()
        fig.savefig(buf,format="png")
        buf.seek(0)
        img_bytes = base64.b64encode(buf.read())
        #b64_string = img_bytes.decode('utf-8')
        #FigureCanvas(fig).print_png(buf)
        received_data['y_hat'] = int(y_hat)
        received_data['img'] = img_bytes.decode('utf-8')
        received_data['y_prob'] = str(y_prob)
        received_data['words'] = []
        scores = a.reshape(-1)
        normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
        for i, w in enumerate(words):
            if normalized_scores[i] >=0:
                received_data['words'].append({
                    'text':w,
                    'size': str(normalized_scores[i])
                })
        #return Response(buf.getvalue(), mimetype='image/png')
        return jsonify(received_data)
        #return render_template('index.html', url ='/static/img/lrp_plot.png')

    return render_template("index.html")

if __name__ == '__main__':

    app.debug = True
    app.run(host='0.0.0.0', port=5566)
