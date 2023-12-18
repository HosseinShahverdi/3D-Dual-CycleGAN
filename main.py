from operator import imod
import os
import glob
import tensorflow as tf
import pre_util as pu
from build_data import data_writer
from solver import Solver
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '1', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_bool('is_cycle_consistent', True, 'cycle-consistent loss for the generator, default: True')
tf.flags.DEFINE_float('cycle_consistent_weight', 10., 'weight for the cycle-consistent loss term, default: 10.')
tf.flags.DEFINE_bool('is_voxel', True, 'voxel-wise loss for the generator, default: True')
tf.flags.DEFINE_float('L1_lambda', 100., 'L1 lambda for conditional voxel-wise loss, default: 100.')
tf.flags.DEFINE_bool('is_gdl', True, 'gradient difference loss (GDL) for the generator, default: True')
tf.flags.DEFINE_float('gdl_weight', 100., 'weight (hyper-parameter) for gradient difference loss term, default: 100.')
tf.flags.DEFINE_bool('is_perceptual', True, 'perceptual loss for the generator, default: True')
tf.flags.DEFINE_float('perceptual_weight', 1., 'weight (hyper-parameter) for perceputal loss term, default: 1.')
tf.flags.DEFINE_integer('perceptual_mode', 5, 'feature layers [1|2|3|4|5], default: 5')
tf.flags.DEFINE_bool('is_ssim', True, 'SSIM loss for the generator, default: True')
tf.flags.DEFINE_float('ssim_weight', 0.05, 'weight (hyper-parameter) for ssim loss term, default: 0.05')
tf.flags.DEFINE_string('dis_model', 'a', 'discriminator model, select from [a|b|c|d|e|f|g], default: a')
tf.flags.DEFINE_string('learning_mode', 'super', 'learning mode, select from [super, unsuper, semi], default, semi')
tf.flags.DEFINE_bool('is_alternative_optim', True, 'optimizing by alterative or integrated optimziation. default: True')
tf.flags.DEFINE_bool('is_lsgan', False, 'use LSGAN loss, default: False')
tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: True')

tf.flags.DEFINE_integer('batch_size', 1, 'batch size for one iteration, default: 1')
tf.flags.DEFINE_string('dataset', 'DC2Anet_db', 'dataset name, default: DC2Anet_db')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial leraning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_integer('iters', 10000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 1000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 50, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_string('load_model', '20221204-0200', 'folder of saved model that you wish to continue training '
                                           '(e.g. 20181127-2116), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train() 
    if not FLAGS.is_train:
        # create nii to predicted nii
        TEST = 'PT'
        idx = 0
        name_patient_list = []
        z_size = []
        sum = 0
        file_name  = glob.glob(os.path.abspath("DC2Anet_db/nifti_sample/CT/*gz"))
        for f in file_name:
            f_split = f.split("\\")
            name_patient = f_split[-1]
            name_patient_list.append(name_patient)
            z = pu.nii_to_sample(name_patient, 'ct', idx)
            z_size.append(z)
            sum += z
            idx += 1
        data_writer(os.path.abspath("dataset/ready_oneSample"),"test")
        solver.test(sum)
        pu.creat_nii(name_patient_list,z_size)
        pu.add_header(name_patient_list)


if __name__ == '__main__':
    tf.compat.v1.app.run()
