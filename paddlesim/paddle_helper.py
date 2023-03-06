# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Authors: drift
Email:   litangshengsheng@baidu.com
Date:    2023/02/10 10:52 PM
"""
import numpy as np  # type: ignore
import os
import sys
import logging

from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar, Any
import copy

from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Program, Variable, Operator


def is_orphan_blob(op_type, port_key):
    skip_dict = {
        'batch_norm': ['MeanOut', 'SavedMean', 'SavedVariance', 'VarianceOut'],
        'dropout': ['Mask'],
        'flatten_contiguous_range': ['XShape'],
        'flatten2': ['XShape'],
        'reshape2': ['XShape'],
        'transpose2': ['XShape'],
        'norm': ['Norm'],
    }
    if skip_dict.get(op_type) is not None and port_key in skip_dict[op_type]:
        return True
    else:
        return False


def flatten_program_blocks(inference_program: Program):
    tensor_dicts: Dict[str:Variable] = dict()
    nodes: List[Operator] = list()
    for cur_block in inference_program.blocks:
        tensor_dicts.update(cur_block.vars)
        nodes.extend(cur_block.ops)
    return tensor_dicts, nodes


def flatten_node_inout_tensors(node: Operator):
    input_names: List[str] = []
    output_names: List[str] = []
    for key in node.input_names:
        for tensor_name in node.input(key):
            assert tensor_name not in input_names, 'duplicated tensor in both input port, currently this is illegal!'
            input_names.append(tensor_name)

    for key in node.output_names:
        if not is_orphan_blob(node.type, key):
            for tensor_name in node.output(key):
                assert tensor_name not in output_names, \
                    'duplicated tensor in both output port, currently this is illegal!'
                output_names.append(tensor_name)

    return input_names, output_names


def is_dynamic_shape(tensor: Variable):
    if tensor.desc.has_attr('shape'):
        shape = tensor.shape
    else:
        return False   # mainly because feed
    logging.info('tensor info: node({}) shape({})'.format(tensor.name, shape))
    if len(shape) > 1:
        if -1 in shape[1:]:
            return True
        else:
            return False
    else:
        return False


def is_const_tensor(tensor: Variable):
    if tensor.name.startswith('feed') or tensor.name.startswith('fetch'):
        # feed and fetch is special vars type, do not have shape/dtype/...
        return False
    elif tensor.persistable:
        return True
    else:
        return False


def add_features_to_output(inference_program: Program, fetch_target_names: List[str]) -> None:
    """
    append fetch ops to tensors you want to get from network output
    :param inference_program:
    :param fetch_target_names:
    """
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name='fetch',
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def recover_features_to_output(inference_program: Program, fetch_target_names: List[str]) -> None:
    """
    remove those extra output vars
    :param inference_program:
    :param fetch_target_names:
    """
    # final op is all fetch(used append)
    for cur_block in inference_program.blocks:
        while True:
            has_removed = False
            for node in cur_block.ops:
                node: Operator
                if node.type == 'fetch' and (node.input('X')[0] not in fetch_target_names):
                    cur_block._remove_op(node.idx)
                    has_removed = True
            if has_removed:
                continue
            else:
                break


def get_input_tensor(inference_program: Program) -> List[Variable]:
    """
    get network input vars(assume feed node have only one output)
    :return:
    """
    tensor_list = []
    tensor_dicts, nodes = flatten_program_blocks(inference_program)
    for node in nodes:
        if node.type == 'feed':
            name = node.output('Out')[0]
            tensor = tensor_dicts[name]
            tensor_list.append(tensor)
    return tensor_list


def get_output_tensor(inference_program: Program) -> List[Variable]:
    """
    get network output vars(assume fetch node have only one input)
    :return:
    """
    tensor_list = []
    tensor_dicts, nodes = flatten_program_blocks(inference_program)
    for node in nodes:
        if node.type == 'fetch':
            name = node.input('X')[0]
            tensor = tensor_dicts[name]
            tensor_list.append(tensor)
    return tensor_list


def forward(inference_program: Program,
            input_data: Dict[str, np.ndarray] = None,
            input_shapes: Dict[str, List[int]] = None) -> Dict[str, np.ndarray]:
    """
    forward paddle static program, support auto random data forward
    :param inference_program:
    :param input_data:
    :param input_shapes:
    :return:
    """

    def get_input_info():
        tensor_lists = get_input_tensor(inference_program)
        input_dict = {}
        for tensor in tensor_lists:
            input_dict[tensor.name] = {'shape': tensor.shape, 'dtpye': tensor.dtype}
        return input_dict

    input_data = input_data if input_data else {}
    input_shapes = input_shapes if input_shapes else {}

    # auto check input dummy data
    if len(input_shapes) == 0:
        input_shapes = {name: info['shape'] for name, info in get_input_info().items()}
    if len(input_data) == 0:
        for name, shape in input_shapes.items():
            value = np.random.rand(*shape).astype('float32')
            input_data[name] = value
            print(value.shape)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    tensor_lists = get_output_tensor(inference_program)
    results = exe.run(inference_program, feed=input_data, fetch_list=tensor_lists)

    new_results = dict()
    assert len(tensor_lists) == len(results), 'fetch var and infer output mismatch!'
    for idx, tensor in enumerate(tensor_lists):
        new_results[tensor.name] = results[idx]
        # process some const node output have -1 in shape
        assert len(tensor.shape) == len(results[idx].shape), 'fetch var and infer output mismatch!'
        for dim in range(len(tensor.shape)):
            assert int(tensor.shape[dim]) == -1 or int(tensor.shape[dim]) == results[idx].shape[dim]
        # print(tensor.shape, results[idx].shape)

    return new_results


def infer_shape(program, input_shape_dict={}):
    """

    :param program:
    :param input_shape_dict:
    :return:
    """

    OP_WITHOUT_KERNEL_SET = {
        'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
        'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
        'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
        'gen_bkcl_id', 'c_gen_bkcl_id', 'gen_nccl_id', 'c_gen_nccl_id',
        'c_comm_init', 'c_sync_calc_stream', 'c_sync_comm_stream',
        'queue_generator', 'dequeue', 'enqueue', 'heter_listen_and_serv',
        'c_wait_comm', 'c_wait_compute', 'c_gen_hccl_id', 'c_comm_init_hccl',
        'copy_cross_scope'
    }

    var_dict, nodes = flatten_program_blocks(program)
    for name, shape in input_shape_dict.items():
        if name in var_dict:
            var_dict[name].desc.set_shape(shape)

    for i in range(len(program.blocks)):
        for j in range(len(program.blocks[0].ops)):
            if program.blocks[i].ops[j].type in OP_WITHOUT_KERNEL_SET:
                continue
            program.blocks[i].ops[j].desc.infer_shape(program.blocks[i].desc)


def is_program_equal(program_x: Program, program_y: Program):
    """
    tell two inference program is equal
        (by compare set(var.name) and count(nodes.type)
    :param program_x:
    :param program_y:
    :return:
    """
    def get_nodes_count(nodes):
        count_dict = {}
        for node in nodes:
            if node.type not in count_dict:
                count_dict[node.type] = 1
            else:
                count_dict[node.type] += 1
        return count_dict

    var_dict_x, nodes_x = flatten_program_blocks(program_x)
    var_dict_y, nodes_y = flatten_program_blocks(program_y)
    count_x = get_nodes_count(nodes_x)
    count_y = get_nodes_count(nodes_y)

    var_equal = (set(var_dict_x.keys()) == set(var_dict_y.keys()))
    node_equal = count_x == count_y

    return var_equal and node_equal
