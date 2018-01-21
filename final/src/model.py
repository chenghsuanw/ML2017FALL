#testing version of the model

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

batch_size = 64
char_limit = 16
dropout_rate = 0.5
h_size = 75
char_dim = 8
char_hidden = 100
d_layer=2


def affine(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        if use_bias:
            b = tf.get_variable("b", [hidden], initializer=tf.constant_initializer(0.))

        flat_inputs = tf.reshape(inputs, [-1, inputs.get_shape().as_list()[-1]])
        W = tf.get_variable("W", [inputs.get_shape().as_list()[-1], hidden])
        res = tf.matmul(flat_inputs, W)
        
        if use_bias:
            res = tf.nn.bias_add(res, b)
        
        res = tf.reshape(res, [tf.shape(inputs)[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden])
        return res

def dropout(args, dropout_rate, is_train, mode="recurrent"):
    if dropout_rate < 1.0:
        if mode == "embedding":
            return  tf.cond(is_train, lambda: tf.nn.dropout(
                args, dropout_rate, noise_shape=[tf.shape(args)[0], 1]) * dropout_rate, lambda: args)
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            return  tf.cond(is_train, lambda: tf.nn.dropout(
                args, dropout_rate, noise_shape=[tf.shape(args)[0], 1, tf.shape(args)[-1]]) * 1.0, lambda: args)
    
    return args

class gru:

    def __init__(self, num_layers, num_units, input_size, dropout_rate=1.0, is_train=None):
        self.features = {}
        features = ["Gru","Para","Init","Drop"]
        self.features["Count"] = num_layers
        for f in features:
            self.features[f] = []
        
        for layer in range(num_layers):
            
            if layer == 0:
                input_size_ = input_size
            else:
                input_size_ = num_units * 2
            self.features["Gru"].append((tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_), tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_), ))
            self.features["Para"].append((tf.Variable(tf.random_uniform(
                [self.features["Gru"][-1][0].params_size()], -0.1, 0.1), validate_shape=False), tf.Variable(tf.random_uniform(
                [self.features["Gru"][-1][1].params_size()], -0.1, 0.1), validate_shape=False), ))
            self.features["Init"].append((tf.Variable(tf.zeros([1, batch_size, num_units])), tf.Variable(tf.zeros([1, batch_size, num_units])), ))
            ds = []
            for _ in range(d_layer):
                ones = tf.ones([1, batch_size, input_size_], dtype=tf.float32)
                ds.append(dropout(ones,dropout_rate=dropout_rate, is_train=is_train, mode=None))
            self.features["Drop"].append((ds[0],ds[1],))
            
    def __call__(self, inputs, seq_len, dropout_rate=1.0, is_train=None):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        
        for layer in range(self.features["Count"]):
            lul = []
            with tf.variable_scope("fw"):
                out_fw, _ = self.features["Gru"][layer][0](outputs[-1] * self.features["Drop"][layer][0], self.features["Init"][layer][0], self.features["Para"][layer][0])
                lul.append(out_fw)
            with tf.variable_scope("bw"):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * self.features["Drop"][layer][1], seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = self.features["Gru"][layer][1](inputs_bw, self.features["Init"][layer][1], self.features["Para"][layer][1])
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                lul.append(out_bw)
            outputs.append(tf.concat(lul, axis=2))
        return tf.transpose(tf.concat(outputs[1:], axis=2), [1, 0, 2])

class final_pointer:
    def __init__(self, hidden, is_train=None):
        self.features = {}
        self.features["Gru"] = GRUCell(hidden)
        self.features["TMode"] = is_train
        self.features["Drop"] = dropout(tf.ones(
            [batch_size, hidden], dtype=tf.float32), dropout_rate=dropout_rate, is_train=self.features["TMode"])
        
    def __call__(self, init, match, d, mask):
        with tf.variable_scope("ptr_net"):
            result = []
            inp, l1 = pointer(dropout(match, dropout_rate=dropout_rate,
                              is_train=self.features["TMode"]), init * self.features["Drop"], d, mask)
            result.append(l1)
            d_inp = dropout(inp, dropout_rate=dropout_rate,
                            is_train=self.features["TMode"])
            _, state = self.features["Gru"](d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, l2 = pointer(dropout(match, dropout_rate=dropout_rate,
                              is_train=self.features["TMode"]), state * self.features["Drop"], d, mask)
            result.append(l2)
            return result[0], result[1]




def pointer(inputs, state, hidden, mask):
    with tf.variable_scope("pointer"): 
        s = -1e30 * (1 - tf.cast(mask, tf.float32)) 
        temp = tf.tile(tf.expand_dims(state, axis=1), [1, tf.shape(inputs)[1], 1])
        temp = tf.concat([temp, inputs], axis=2)
        s += tf.squeeze(affine(tf.nn.tanh(affine(temp, hidden, use_bias=False, scope="s0")), 1, use_bias=False, scope="s"), [2])
        return tf.reduce_sum(tf.expand_dims(tf.nn.softmax(s), axis=2) * inputs, axis=1), s

def summ(memory, hidden, mask, dropout_rate=1.0, is_train=None):
    with tf.variable_scope("summ"):  
        s = -1e30 * (1 - tf.cast(mask, tf.float32)) + tf.squeeze(affine(tf.nn.tanh(affine(dropout(memory, dropout_rate=dropout_rate, is_train=is_train), hidden, scope="s0")), 1, use_bias=False, scope="s"), [2]) 
        
        return tf.reduce_sum(tf.expand_dims(tf.nn.softmax(s), axis=2) * memory, axis=1)


def attention(inputs, memory, mask, hidden, dropout_rate=1.0, is_train=None):
    with tf.variable_scope("dot_attention"):

        with tf.variable_scope("attention"):
            
            outputs = tf.matmul(tf.nn.relu(
                affine(dropout(inputs, dropout_rate=dropout_rate, is_train=is_train), hidden, use_bias=False, scope="inputs")), tf.transpose(
                tf.nn.relu(
                affine(dropout(memory, dropout_rate=dropout_rate, is_train=is_train), hidden, use_bias=False, scope="memory")), [0, 2, 1])) / (hidden ** 0.5)
            masked = tf.expand_dims(mask, axis=1)
            logits = tf.nn.softmax(-1e30 * (1 - tf.cast(tf.tile(masked, [1, tf.shape(inputs)[1], 1]), tf.float32)) + outputs)
            res = tf.concat([inputs, tf.matmul(logits, memory)], axis=2)

        with tf.variable_scope("gate"):  
            gate = tf.nn.sigmoid(affine(dropout(res, dropout_rate=dropout_rate, is_train=is_train), res.get_shape().as_list()[-1], use_bias=False))
            return res * gate

class RNet(object):
    
    global batch_size, h_size, char_dim, char_hidden
    
    def __init__(self, data, wm, cm):
        
        
        self.set_variable(data, wm, cm)
        c,q = self.emb()
        c,q = self.encodeing(c,q)
        att = self.att(c,q)
        match = self.match(att)
        logits1, logits2 = self.pointer(match,q)
        self.pred(logits1,logits2)
        
        print("******** Start prediction ********")
            

    def emb(self): 
        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                c_context_embs = []
                c_question_embs = []
                c_context_embs.append(tf.reshape(tf.nn.embedding_lookup(
                    self.embeddings[1], self.context_hidden), [batch_size * tf.reduce_max(self.context_length), 16, char_dim]))
                c_question_embs.append(tf.reshape(tf.nn.embedding_lookup(
                    self.embeddings[1], self.question_hidden), [batch_size * tf.reduce_max(self.question_length), 16, char_dim]))
                 
                c_context_embs.append(dropout(
                    c_context_embs[0], dropout_rate=dropout_rate, is_train=self.is_train))
                c_question_embs.append(dropout(
                    c_question_embs[0], dropout_rate=dropout_rate, is_train=self.is_train))
                cell = []
                for _ in range(2):
                    cell.append(tf.contrib.rnn.GRUCell(char_hidden))
                
                _, state = tf.nn.bidirectional_dynamic_rnn(
                    cell[0], cell[1], c_context_embs[1], self.ch_len, dtype=tf.float32)
                c_context_embs.append(tf.concat(state, axis=1))


                _, state = tf.nn.bidirectional_dynamic_rnn(
                    cell[0], cell[1], c_question_embs[1], self.question_hidden_len, dtype=tf.float32)
                question_hidden_emb = tf.concat(state, axis=1)
                question_hidden_emb = tf.reshape(question_hidden_emb, [batch_size, tf.reduce_max(self.question_length), 2 * char_hidden])
                context_hidden_emb = tf.reshape(c_context_embs[2], [batch_size, tf.reduce_max(self.context_length), 2 * char_hidden])
            #print("word embedding done!")
            with tf.name_scope("word"):
                ce = tf.nn.embedding_lookup(self.embeddings[0], self.context)
                qe = tf.nn.embedding_lookup(self.embeddings[0], self.question)
            #print("char embedding done!")
            context_emb = tf.concat([tf.nn.embedding_lookup(self.embeddings[0], self.context), context_hidden_emb], axis=2)
            question_emb = tf.concat([tf.nn.embedding_lookup(self.embeddings[0], self.question), question_hidden_emb], axis=2) 
        #print("all embedding done!")
        return context_emb, question_emb 
    
    def encodeing(self,context_emb,question_emb): 
        
        with tf.variable_scope("encoding"):
            layers = []
            layers.append(gru(num_layers=3, num_units=h_size, input_size=context_emb.get_shape(
            ).as_list()[-1], dropout_rate=dropout_rate, is_train=self.is_train))
            layers.append(layers[0](context_emb, seq_len=self.context_length))
            layers.append(layers[0](question_emb, seq_len=self.question_length))
        #print("encoding done!")
        self.encodeing_layers = layers
        return layers[1], layers[2]

    def att(self,c,q):
        with tf.variable_scope("attention"):
            layers = []
            layers.append(attention(c, q, mask=self.question_mask, hidden=h_size,
                                   dropout_rate=dropout_rate, is_train=self.is_train)) 
            layers.append(gru(num_layers=1, num_units=h_size, input_size=layers[0].get_shape(
            ).as_list()[-1], dropout_rate=dropout_rate, is_train=self.is_train)) 
            layers.append(layers[1](layers[0], seq_len=self.context_length))
        #print("att done!")
        self.att_layers = layers
        return layers[2]

    def match(self,att):
        with tf.variable_scope("match"):
            layers = []
            layers.append(attention(
                att, att, mask=self.context_mask, hidden=h_size, dropout_rate=dropout_rate, is_train=self.is_train))
            
            layers.append(gru(num_layers=1, num_units=h_size, input_size=layers[0].get_shape(
            ).as_list()[-1], dropout_rate=dropout_rate, is_train=self.is_train))
            layers.append(layers[1](layers[0], seq_len=self.context_length))
        self.match_layer = layers
        #print("match done!")
        return layers[2]

    def pointer(self,match,q):
        with tf.variable_scope("pointer"):
            layers = []
            layers.append(summ(q[:, :, -2 * h_size:], h_size, mask=self.question_mask,
                        dropout_rate=dropout_rate, is_train=self.is_train))
            layers.append(final_pointer(hidden=layers[0].get_shape().as_list(
            )[-1], is_train=self.is_train))
            
            logits1, logits2 = layers[1](layers[0], match, h_size, self.context_mask)
        #print("pointer done!")
        self.pointer_layer = layers
        return logits1, logits2

    def pred(self,l1,l2):
        with tf.variable_scope("predict"):
            
            self.ys = []
            for i in range(2):
                n1 = tf.nn.softmax(l1)
                n2 = tf.nn.softmax(l2)
                self.ys.append(tf.argmax(tf.reduce_max(tf.matrix_band_part(tf.matmul(tf.expand_dims(n1, axis=2),
                    tf.expand_dims(n2, axis=1)), 0, 15), axis=2-i), axis=1))
            
            loss = []
            loss.append(tf.nn.softmax_cross_entropy_with_logits(
                logits=l1, labels=self.x))
            loss.append(tf.nn.softmax_cross_entropy_with_logits(
                logits=l2, labels=self.y))
            
            loss = tf.add(loss[0],loss[1])
            self.loss = tf.reduce_mean(loss)
        #print("pred done!")
    

    def set_variable(self,data,wm,cm):
        two_zeros = [0,0]
        three_zeros = [0,0,0]
        context, question, context_hidden, question_hidden, x, y, self.qa_id = data.get_next()
        
        
        self.context_length = tf.reduce_sum(tf.cast(tf.cast(context, dtype = tf.bool), dtype = tf.int32), axis=1)
        self.question_length = tf.reduce_sum(tf.cast(tf.cast(question, dtype =tf.bool), dtype = tf.int32), axis=1)

        self.context = tf.slice(context, two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        self.question = tf.slice(question, two_zeros, [batch_size, tf.reduce_max(self.question_length)])

        self.context_mask = tf.slice(tf.cast(context, dtype = tf.bool), two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        self.question_mask = tf.slice(tf.cast(question, dtype =tf.bool), two_zeros, [batch_size, tf.reduce_max(self.question_length)])

        self.context_hidden = tf.slice(context_hidden, three_zeros, [batch_size, tf.reduce_max(self.context_length), 16])
        self.question_hidden = tf.slice(question_hidden,three_zeros, [batch_size, tf.reduce_max(self.question_length), 16])
        
        self.x = tf.slice(x, two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        self.y = tf.slice(y, two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        lengths = [tf.slice(question_hidden,three_zeros, [batch_size, tf.reduce_max(self.question_length), 16]),
            tf.slice(context_hidden, three_zeros, [batch_size, tf.reduce_max(self.context_length), 16])]
        l = []
        for k in lengths:
            l.append(tf.reduce_sum(tf.cast(tf.cast(k, tf.bool), tf.int32), axis=2))
        self.ch_len = tf.reshape(l[1], [-1])
        self.question_hidden_len = tf.reshape(l[0], [-1])

        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.embeddings = []
        self.embeddings.append(tf.get_variable("word_mat", initializer=tf.constant(
            wm, dtype=tf.float32),trainable=False))
        self.embeddings.append(tf.get_variable(
            "char_mat", initializer=tf.constant(cm, dtype=tf.float32),trainable=False))
        
    
    def get_loss(self):
        return self.loss

    

