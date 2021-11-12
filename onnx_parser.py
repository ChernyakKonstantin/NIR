import json

import numpy as np
import onnx
from google.protobuf.json_format import MessageToJson
from onnx import shape_inference, TensorProto, ValueInfoProto


class Parser:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.onnx_model = onnx.load(self.model_path)
        self.inferred_model = shape_inference.infer_shapes(self.onnx_model)

    def parse(self) -> tuple:
        s = MessageToJson(self.onnx_model)
        onnx_json = json.loads(s)
        adj_mat = self.get_adj_mat(onnx_json)
        op_type = self.get_op_types(onnx_json)
        in_shapes = self.get_input_shapes(onnx_json)
        out_shapes = self.get_output_shapes(onnx_json)
        return op_type, in_shapes, out_shapes, adj_mat

    def get_input_shapes(self, onnx_json: dict) -> list:
        in_shapes = []
        for node in onnx_json['graph']['node']:
            in_shapes_ = []
            for input_name in node['input']:
                input_info = (list(filter(lambda x: x.name == input_name, self.inferred_model.graph.value_info))
                              + list(filter(lambda x: x.name == input_name, self.inferred_model.graph.input))
                              + list(filter(lambda x: x.name == input_name, self.inferred_model.graph.initializer)))
                input_spec = next(iter(input_info))
                if isinstance(input_spec, TensorProto):
                    in_shape = list(input_spec.dims)
                elif isinstance(input_spec, ValueInfoProto):
                    in_shape = [int(dim.dim_value) for dim in input_spec.type.tensor_type.shape.dim]
                in_shapes_.append(in_shape)
            in_shapes.append(in_shapes_)
        return self.align_shapes(in_shapes)

    def get_output_shapes(self, onnx_json: dict) -> list:
        out_shapes = []
        for node in onnx_json['graph']['node']:
            out_shapes_ = []
            for output_name in node['output']:
                output_info = (list(filter(lambda x: x.name == output_name, self.inferred_model.graph.value_info))
                               + list(filter(lambda x: x.name == output_name, self.inferred_model.graph.output)))
                output_spec = next(iter(output_info))
                out_shape = [int(dim.dim_value) for dim in output_spec.type.tensor_type.shape.dim]
                out_shapes_.append((out_shape))
            out_shapes.append(out_shapes_)
        return self.align_shapes(out_shapes)

    def get_adj_mat(self, onnx_json: dict) -> np.ndarray:
        nodes = onnx_json['graph']['node']
        num_nodes = len(nodes)
        adj = np.zeros((num_nodes, num_nodes), dtype='int')
        # j is for row, i is for column
        for i, node_1 in enumerate(nodes):
            outputs: list = node_1['output']
            for j, node_2 in enumerate(nodes):
                inputs: list = node_2['input']
                for output in outputs:
                    if output in inputs:
                        adj[j, i] = 1
        return adj

    def get_op_types(self, onnx_json: dict) -> list:
        return [node['opType'] for node in onnx_json['graph']['node']]

    def align_shapes(self, shapes: list) -> list:
        max_nodes = max([len(shape) for shape in shapes])
        tmp = []
        [tmp.extend(shape) for shape in shapes]
        max_dim = max([len(shape) for shape in tmp])
        # Выровнять размерности
        for i in range(len(shapes)):
            for j in range(len(shapes[i])):
                n_absent_dim = max_dim - len(shapes[i][j])
                tmp = [0 for _ in range(n_absent_dim)]
                tmp.extend((shapes[i][j]))
                shapes[i][j] = tmp
        # Выровнять число входных нод
        for i in range(len(shapes)):
            n_absent_nodes = max_nodes - len(shapes[i])
            for _ in range(n_absent_nodes):
                shapes[i].append([0] * max_dim)
        return shapes
        
