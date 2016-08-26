#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core functions of TV."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import numpy as np
import tensorflow as tf

import tensorvision.utils as utils


def load_weights(checkpoint_dir, sess, saver):
    """
    Load the weights of a model stored in saver.

    Parameters
    ----------
    checkpoint_dir : str
        The directory of checkpoints.
    sess : tf.Session
        A Session to use to restore the parameters.
    saver : tf.train.Saver

    Returns
    -----------
    int
        training step of checkpoint
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])


def build_training_graph(hypes, queue, modules):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    queue: tf.queue
        Data Queue
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """

    data_input = modules['input']
    encoder = modules['arch']
    objective = modules['objective']
    optimizer = modules['solver']

    learning_rate = tf.placeholder(tf.float32)

    # Add Input Producers to the Graph
    with tf.name_scope("Inputs"):
        image, labels = data_input.inputs(hypes, queue, 'train')

    # Run inference on the encoder network
    logits = encoder.inference(hypes, image, train=True)

    # Build decoder on top of the logits
    decoded_logits = objective.decoder(hypes, logits, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = objective.loss(hypes, decoded_logits,
                                labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        train_op = optimizer.training(hypes, losses,
                                      global_step, learning_rate)

    with tf.name_scope("Evaluation"):
        # Add the Op to compare the logits to the labels during evaluation.
        eval_list = objective.evaluation(
            hypes, image, labels, decoded_logits, losses, global_step)

        summary_op = tf.merge_all_summaries()

    graph = {}
    graph['losses'] = losses
    graph['eval_list'] = eval_list
    graph['summary_op'] = summary_op
    graph['train_op'] = train_op
    graph['global_step'] = global_step
    graph['learning_rate'] = learning_rate

    return graph


def build_inference_graph(hypes, modules, image):
    """Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    modules : tuble
        the modules load in utils
    image : placeholder

    return:
        graph_ops
    """
    with tf.name_scope("Validation"):

        tf.get_variable_scope().reuse_variables()

        logits = modules['arch'].inference(hypes, image, train=False)

        decoded_logits = modules['objective'].decoder(hypes, logits,
                                                      train=False)
    return decoded_logits


def start_tv_session(hypes):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    tuple
        (sess, saver, summary_op, summary_writer, threads)
    """
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    if 'keep_checkpoint_every_n_hours' in hypes['solver']:
        kc = hypes['solver']['keep_checkpoint_every_n_hours']
    else:
        kc = 10000.0
    saver = tf.train.Saver(max_to_keep=int(utils.cfg.max_to_keep),
                           keep_checkpoint_every_n_hours=kc)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(hypes['dirs']['output_dir'],
                                            graph=sess.graph)

    tv_session = {}
    tv_session['sess'] = sess
    tv_session['saver'] = saver
    tv_session['summary_op'] = summary_op
    tv_session['writer'] = summary_writer
    tv_session['coord'] = coord
    tv_session['threads'] = threads

    return tv_session


def do_eval(hypes, eval_list, phase, sess):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    eval_list : list of tuples
        Each tuple should contain a string (name if the metric) and a
        tensor (storing the result of the metric).
    phase : str
        Describes the data the evaluation is run on.
    sess : tf.Session
        The session in which the model has been trained.

    Returns
    -------
    tuple of lists
        List of names and evaluation results
    """
    # And run one epoch of eval.
    # Checking for List for compability
    if eval_list[phase] is None:
        return [''], [0.0]
    if type(eval_list[phase]) is list:
        eval_names, eval_op = zip(*eval_list[phase])

    else:
        logging.warning("Passing eval_op directly is deprecated. "
                        "Pass a list of tuples instead.")
        eval_names = ['Accuracy']
        eval_op = [eval_list[phase]]

    assert(len(eval_names) == len(eval_op))

    if phase == 'train':
        num_examples = hypes['data']['num_examples_per_epoch_for_train']
    if phase == 'val':
        num_examples = hypes['data']['num_examples_per_epoch_for_eval']

    steps_per_epoch = num_examples // hypes['solver']['batch_size']
    num_examples = steps_per_epoch * hypes['solver']['batch_size']

    logging.info('Data: % s  Num examples: % d ' % (phase, num_examples))
    # run evaluation on num_examples many images
    results = sess.run(eval_op)
    logging.debug('Output of eval: %s', results)
    for step in xrange(1, steps_per_epoch):
        results = map(np.add, results, sess.run(eval_op))

    avg_results = [result / steps_per_epoch for result in results]

    for name, value in zip(eval_names, avg_results):
        logging.info('%s : % 0.04f ' % (name, value))

    return eval_names, avg_results
