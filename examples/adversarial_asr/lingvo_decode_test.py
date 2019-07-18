import tensorflow as tf
from lingvo import model_imports
from lingvo import model_registry
import numpy as np
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool_decode import Transform, create_features, create_inputs
import time
from lingvo.core import cluster_factory
from absl import flags
from absl import app
from os import path

# data directory

flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string('input', 'L2-Arctic.txt',
                    'Input audio .wav file(s), at 16KHz (separated by spaces)')
flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                    'location of checkpoint')
flags.DEFINE_integer('num_gpu', '0', 'which gpu to run')
flags.DEFINE_integer('batch_size', '5', 'batch size')
FLAGS = flags.FLAGS


def ReadFromWav(data, batch_size):
    """
    Returns:
        audios_np: a numpy array of size (batch_size, max_length) in float
        trans: a numpy array includes the targeted transcriptions (batch_size, )
        max_length: the max length of the batch of audios
        sample_rate_np: a numpy array
        lengths: a list of the length of original audios
    """
    audios = []
    lengths = []

    # read the .wav file
    for i in range(batch_size):
        if path.isabs(str(data[0, i])):
            sample_rate_np, audio_temp = wav.read(str(data[0, i]))
        else:
            sample_rate_np, audio_temp = wav.read(FLAGS.root_dir + str(data[0, i]))
        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp
        length = len(audio_np)
        audios.append(audio_np)
        lengths.append(length)

    max_length = max(lengths)

    # pad the input audio
    audios_np = np.zeros([batch_size, max_length])
    for i in range(batch_size):
        audio_float = audios[i].astype(float)
        audios_np[i, :lengths[i]] = audio_float

    # read the transcription
    trans = data[1, :]

    return audios_np, trans,max_length, sample_rate_np, lengths


class DecodeL:
    def __init__(self, sess,batch_size):
        self.sess = sess
        self.batch_size=batch_size

        tf.set_random_seed(1234)
        params = model_registry.GetParams('asr.librispeech.Librispeech960Wpm', 'Test')
        params.random_seed = 1234
        params.is_eval = True
        params.cluster.worker.gpus_per_replica = 1
        cluster = cluster_factory.Cluster(params.cluster)
        with cluster, tf.device(cluster.GetPlacer()):
            model = params.cls(params)

            # placeholders
            self.input_tf = tf.placeholder(tf.float32, shape=[batch_size, None], name='qq_input')
            self.tgt_tf = tf.placeholder(tf.string)
            self.sample_rate_tf = tf.placeholder(tf.int32, name='qq_sample_rate')
            self.maxlen = tf.placeholder(np.int32)

            # generate the inputs that are needed for the lingvo model
            self.features = create_features(self.input_tf, self.sample_rate_tf)
            self.inputs = create_inputs(model, self.features, self.tgt_tf, self.batch_size)

            task = model.GetTask()
            metrics = task.FPropDefaultTheta(self.inputs)
            self.decoded = task.Decode(self.inputs)

    def decode_stage1(self,audios,trans,maxlen,sample_rate):
        sess = self.sess
        sess.run(tf.initializers.global_variables())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)
        feed_dict = {self.input_tf: audios,
                     self.tgt_tf: trans,
                     self.sample_rate_tf: sample_rate,
                     self.maxlen:maxlen}
        #writer=tf.summary.FileWriter("/home/ubuntu/adv/demo")
        #writer.add_graph(sess.graph)
        predictions = sess.run(self.decoded, feed_dict)
        # show the initial predictions
        pred=[]
        refText=[]
        for i in range(len(audios)):
            pred.append(predictions['topk_decoded'][i, 0].upper())
            refText.append(trans[i])
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("refText:{}".format(trans[i].lower()))
        return pred,refText





def main(argv):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    batches=int(len(data[0])/10)
    data = data[:, FLAGS.num_gpu * 10: (FLAGS.num_gpu + batches) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = num / batch_size
    assert num % batch_size == 0

    with tf.device("/gpu:0"):
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess:
            # set up the attack class
            decodel = DecodeL(sess,batch_size)
            all_pred=[]
            all_refText=[]
            for l in range(num_loops):
                print("=========== current loop",str(l))
                data_sub = data[:, l * batch_size:(l + 1) * batch_size]

                # stage 1
                # all the output are numpy arrays
                audios, trans, maxlen, sample_rate,lengths = ReadFromWav(
                    data_sub, batch_size)
                # Loading data: data stored in audios trans
                pred,refText=decodel.decode_stage1(audios, trans,maxlen, sample_rate)
                all_pred=all_pred+pred
                all_refText=all_refText+refText
            with open(FLAGS.input+".pred", 'w') as fw:
                fw.write("\n".join(item for item in all_pred))
            with open(FLAGS.input+".refText", 'w') as fw:
                fw.write("\n".join(item for item in all_refText))

if __name__ == '__main__':
    app.run(main)




