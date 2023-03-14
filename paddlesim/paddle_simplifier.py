# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Authors: drift
Email:   litangshengsheng@baidu.com
Date:    2023/02/10 10:52 PM
"""
import os
import sys
import logging
from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar
import numpy as np  # type: ignore

import paddle
from paddle import fluid
from paddle.fluid.framework import Program, Variable, Operator
from paddle.fluid.framework import convert_np_dtype_to_dtype_

import paddlesim.paddle_helper as ph


T = TypeVar('T')
already_processed_name = []


def get_constant_nodes(inference_program: Program) -> List[Operator]:
    """

    :param inference_program:
    :return:
    """
    def process_constant_node():
        if len(input_names) == 0 or len(output_names) == 0:
            # special node process: feed(input)、fetch(output)、fill(const)
            # those node have only output/input
            if node.type in ['fetch', 'feed']:
                return
            elif 'fill_' in node.type or 'assign_value' in node.type:
                for output_name in output_names:
                    const_tensor_names.append(output_name)
                    const_node_lists.append(node)
            else:
                print(node.type)

        elif node.type == 'shape':
            # consider shape as const node
            # TODO consider dynamic H/W vs dynamic batch
            for output_name in output_names:
                const_tensor_names.append(output_name)
                const_node_lists.append(node)

        elif any(x in dynamic_tensor_names for x in input_names):
            dynamic_tensor_names.extend(output_names)
            # TODO there will be fix-output shape layers

        elif all([x in const_tensor_names for x in input_names]):
            for output_name in output_names:
                const_tensor_names.append(output_name)
                const_node_lists.append(node)
        else:
            # means this node keep its dynamic & const & tensor status
            return

    tensor_dicts, nodes = ph.flatten_program_blocks(inference_program)
    const_tensor_names: List[str] = []
    dynamic_tensor_names: List[str] = []
    const_node_lists: List[Operator] = []

    # init const & dynamic tensor
    for name, tensor in tensor_dicts.items():
        if ph.is_const_tensor(tensor):
            const_tensor_names.append(name)
        else:
            if ph.is_dynamic_shape(tensor):
                dynamic_tensor_names.append(name)

    # The output shape of some node types is determined by the input value
    # we consider the output of this node doesn't have constant shape,
    # so we do not simplify a such node even if the node is Shape op
    for node in nodes:
        input_names, output_names = ph.flatten_node_inout_tensors(node)
        process_constant_node()

    info = [node.output_arg_names for node in const_node_lists]
    new_info = []
    flatten_info = []
    for nodes_name in info:
        for node_name in nodes_name:
            if node_name not in already_processed_name:
                new_info.append(node_name)
            flatten_info.append(node_name)
    already_processed_name.extend(flatten_info)
    print('cur step get newly const tensor: {}'.format(new_info))

    return const_node_lists


def forward_for_node_outputs(inference_program: Program,
                             nodes: List[Operator],
                             input_data: Dict[str, np.ndarray] = None,
                             input_shapes: Dict[str, List[int]] = None) -> Dict[str, np.ndarray]:
    output_extra_list = []
    for node in nodes:
        _, output_names = ph.flatten_node_inout_tensors(node)
        output_extra_list.extend(output_names)

    ph.add_features_to_output(inference_program, output_extra_list)
    res = ph.forward(inference_program,
                     input_data=input_data,
                     input_shapes=input_shapes)
    ph.recover_features_to_output(inference_program, output_extra_list)

    return res


def eliminate_const_nodes(inference_program: Program, const_nodes: List[Operator],
                          res: Dict[str, np.ndarray]):
    """
    :param inference_program: the original paddle program
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: the simplified onnx model. Redundant ops are all removed.
    """
    def change_var_into_param(c_block, d_node):
        # print(d_node.output_arg_names)
        for key in d_node.output_names:
            if ph.is_orphan_blob(d_node.type, key):
                continue
            for var_name in d_node.output(key):
                # del old var(this is un-used)
                c_block._remove_var(var_name)
                # re-new vars
                dtype = convert_np_dtype_to_dtype_(res[var_name].dtype)
                _ = c_block.create_var(name=var_name,
                                       shape=res[var_name].shape,
                                       is_data=True,
                                       dtype=dtype)
                # following process referenced from paddle.nn.initializer.Assign
                from paddle.fluid.core import VarDesc
                if dtype == VarDesc.VarType.FP32:
                    value_name = "fp32_values"
                    values = [float(v) for v in res[var_name].flat]
                elif dtype == VarDesc.VarType.INT32:
                    value_name = "int32_values"
                    values = [int(v) for v in res[var_name].flat]
                elif dtype == VarDesc.VarType.INT64:
                    value_name = "int64_values"
                    values = [int(v) for v in res[var_name].flat]
                else:
                    raise ValueError("Unsupported dtype %s", dtype)
                _ = c_block._insert_op(index=d_node.idx+1, type='assign_value',
                                       outputs={'Out': var_name},
                                       attrs={
                                           'dtype': convert_np_dtype_to_dtype_(res[var_name].dtype),
                                           'shape': res[var_name].shape,
                                           value_name: values})
                # print(var_name, res[var_name].shape, convert_np_dtype_to_dtype_(res[var_name].dtype))
                # add new param into graph
                # c_block.vars[var_name] = new_params
                c_block._sync_with_cpp()

                # set param data
                # print(c_block.has_var(new_name))
                # c_block.vars[new_name].set_value(res[var_name])
                # c_block.vars['fill_constant_1.tmp_0'].set_value(np.random.rand(1))
        # del old node
        cur_block._remove_op(d_node.idx)

    for cur_block in inference_program.blocks:
        for i, node in enumerate(cur_block.ops):
            if node in const_nodes:
                change_var_into_param(cur_block, node)


def _infer_shapes_and_optimize(inference_program: Program, input_shape_dict: Dict) -> Program:
    """

    :param inference_program:
    :param input_shape_dict:
    :return:
    """
    # tensor_list = ph.get_output_tensor(inference_program)
    new_program = inference_program._inference_optimize()
    ph.infer_shape(new_program, input_shape_dict)

    return new_program


def shaping_and_optimize(inference_program: Program) -> Program:
    """

    :param inference_program:
    :param input_shape_dict:
    :return:
    """
    # tensor_list = ph.get_output_tensor(inference_program)

    ph.infer_shape(inference_program, {})
    new_program = inference_program._inference_optimize()

    return new_program


def constant_folding(inference_program: Program) -> Program:
    """

    :param inference_program:
    :return:
    """
    # model original output
    tensor_list = ph.get_output_tensor(inference_program)
    # get const node and its data
    const_nodes = get_constant_nodes(inference_program)
    res = forward_for_node_outputs(inference_program, const_nodes)
    # eliminate and recover
    eliminate_const_nodes(inference_program, const_nodes, res)
    ph.recover_features_to_output(inference_program, [tensor.name for tensor in tensor_list])

    new_program = inference_program._inference_optimize()
    # print(inference_program)

    return new_program


def shaping_and_folding_loop(x: T, shaping_func: Callable[[T], T], folding_func: Callable[[T], T]) -> T:
    """
    Run shaping_func + folding_func until paddle program do not change any more
    :param x:
    :param shaping_func:
    :param folding_func:

    :return:
    """
    x = folding_func(x)
    x = shaping_func(x)

    while True:
        y = folding_func(x)
        y = shaping_func(y)
        if ph.is_program_equal(x, y):
            return x


def simplify(model_dir, model_filename, params_filename, input_info, save_dir='output'):
    paddle.enable_static()
    exe = fluid.Executor(fluid.CPUPlace())
    [prog, ipts, outs] = fluid.io.load_inference_model(
        model_dir,
        exe,
        model_filename=model_filename,
        params_filename=params_filename)

    # pre-fix shape to fix input shape
    assert isinstance(input_info, dict)
    prog = _infer_shapes_and_optimize(prog, input_shape_dict=input_info)

    new_program = shaping_and_folding_loop(prog, shaping_and_optimize, constant_folding)

    fluid.io.save_inference_model(save_dir, ipts, outs, exe, new_program,
                                  model_filename=model_filename,
                                  params_filename=params_filename)


if __name__ == '__main__':
    simplify('./VIMER-UFO', 'model.pdmodel', 'model.pdiparams', input_info={'x': (1, 3, 224, 224)})
