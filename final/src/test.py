import tensorflow as tf
import json
import csv
import pickle
import jieba
from model import RNet
import sys



batch_size = 64
w_emb_d = './embeddings/word_emb.pickle'
c_emb_d = './embeddings/char_emb.pickle'
rf = "./data/test.tfrecords"
model_d = "./best_model/model_8000.ckpt"

def parser():
    def parse(example):
        x = {}
        todo = ["context_idxs","ques_idxs","context_char_idxs","ques_char_idxs","y1","y2"]
        for t in todo:
            x[t] = tf.FixedLenFeature([], tf.string)
        x["id"] = tf.FixedLenFeature([], tf.int64)
        
        f = tf.parse_single_example(example,features=x)
        ids = []
        ids.append(tf.reshape(tf.decode_raw(f["context_idxs"], tf.int32), [1000]))
        ids.append(tf.reshape(tf.decode_raw(f["ques_idxs"], tf.int32), [100]))
        ids.append(tf.reshape(tf.decode_raw(f["context_char_idxs"], tf.int32), [1000, 16]))
        ids.append(tf.reshape(tf.decode_raw(f["ques_char_idxs"], tf.int32), [100, 16]))
        ids.append(f["id"])
        ys = []
        ys.append(tf.reshape(tf.decode_raw(f["y1"], tf.float32), [1000])) 
        ys.append(tf.reshape(tf.decode_raw(f["y2"], tf.float32), [1000])) 
             
        return ids[0], ids[1], ids[2], ids[3], ys[0], ys[1], ids[4]
    return parse

def make_example(filename):
    
    
    test_examples = {}
    count = 0
    print()
    with open(filename, "r") as fh:
        source = json.load(fh)
        for topic in source["data"]:
            for p in topic["paragraphs"]:
                context = p["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = [token for token in jieba.cut(context,cut_all=False)]
                
                current = 0
                spans = []
                for token in context_tokens:
                    current = context.find(token, current)
                    spans.append((current, current + len(token)))
                    current = current + len(token)
                for qa in p["qas"]:
                    count += 1
                    
                    starts, ends = [], []
                    answer_texts = []
                    answer_texts.append("123")
                    answer_span = []
                    for idx, _ in enumerate(spans):
                        answer_span.append(idx)
                    
                    starts.append(answer_span[0])
                    ends.append(answer_span[-1])
                    
                    test_examples[str(count)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        
    return test_examples

def main(argv):
    global rf, w_emb_d, c_emb_d, model_d
    
    with open(w_emb_d, 'rb') as handle:
        w_emb = pickle.load(handle)
    with open(c_emb_d, 'rb') as handle:
        c_emb = pickle.load(handle)
    
    test = make_example(argv[0])
    
    model = RNet(tf.data.TFRecordDataset(rf).map(
        parser()).repeat().batch(batch_size).make_one_shot_iterator(), w_emb[1], c_emb[1])

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_d)
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        
        
        final = []
        for _ in range(len(test) // batch_size + 1):
            ys = model.ys
            qa_id, _, y1, y2 = sess.run([model.qa_id, model.loss, ys[0], ys[1]])
            qq = []
            for qid, p1, p2 in zip(qa_id.tolist(), y1.tolist(), y2.tolist()):
                qq.append((test[str(qid)]["uuid"],test[str(qid)]["spans"][p1][0],test[str(qid)]["spans"][p2][1]-1))

            final.append(qq)

        f = open(argv[1], 'w')
        w = csv.writer(f)
        w.writerow(['id','answer']) 
        count = 0   
        for l in final:
            for k in l :
                if count < len(test):
                    w.writerow([k[0], " ".join(str(x) for x in range(k[1],k[2]+1))])
                    print(k)
                count += 1
        f.close()

if __name__ == '__main__':
    main(sys.argv[1:])


