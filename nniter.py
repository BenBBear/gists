import mxnet as mx
import logging
import sys
import json
import random
import string
from collections import namedtuple, defaultdict
from copy import copy


################## Utils #################


def get_params(node, params):
    r = {}
    for x in node['input_nodes']:
        name = ('aux:' + x['name']) if x['is_aux'] else ('arg:' + x['name'])
        if name in params:
            r[name] = params[name]
    return r


def rand_str():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

################## Profiles #################

class Profile:
    def __init__(self):
        self._id = 0
        self.fast_stop = True

    def empty_output(self):
        return { 'next': []}

    def reset_id(self):
        self._id = 0

    def merge(self, curr_output=None, curr_node=None, params=None):
        raise NotImplementedError

    def strip_output(self, n):
        raise NotImplementedError

    def ready(self, curr_output):
        raise NotImplementedError


class NeuronWakeUp(Profile):

    def merge(self, curr_output=None, curr_node=None, params=None):
        if curr_node['op'] is None:
            return curr_output
        opname = curr_node['op'].lower()
        if ('convolution' in opname) or ('fullyconnected' in opname): # collect conv params
            curr_output.update(get_params(curr_node, params))
            curr_output['_id'] = self._id
            self._id += 1
            curr_output['has_conv_weight'] = True
        elif 'batchnorm' in opname: # collect bn params
            if curr_output['has_conv_weight']:
                curr_output.update(get_params(curr_node, params))
                curr_output['has_bn_param'] = True
        else:
            pass # do nothing

        return curr_output

    def ready(self, curr_output):
        return curr_output['has_conv_weight'] and curr_output['has_bn_param']

    def empty_output(self):
        return { 'next': [], 'prev': [], 'has_conv_weight': False, 'has_bn_param': False}

    def strip_output(self, n):
        del n['_id']
        del n['has_conv_weight']
        del n['has_bn_param']
        del n['prev']
        next = []
        for nx in n['next']:
            _nx = {'_id':0, 'has_conv_weight':0, 'has_bn_param':0, 'prev':0,'next':0}
            _nx.update(nx)
            del _nx['_id']
            del _nx['has_conv_weight']
            del _nx['has_bn_param']
            del _nx['prev']
            del _nx['next']
            next.append(_nx)
        del n['next']
        return {
            'params':n,
            'next':next
        }



NNIterProfile = namedtuple("NNIterProfile", "NeuronWakeUp")(
    NeuronWakeUp=NeuronWakeUp(),
    # to be added
)


################## Implementation #################


class nniter:
    def __init__(self, symbol_json_file_path=None,
                 params_file_path=None,
                 profile=NNIterProfile.NeuronWakeUp):
        logging.basicConfig(format="%(name)s ==%(funcName)s-%(levelname)s==> %(message)s")
        self._logger = logging.getLogger('nniter')
        self._logger.addHandler(logging.StreamHandler())
        self._logger.setLevel(logging.INFO)

        self._params = {}
        self._queue = []
        self.current = 0
        self.length = 0
        if params_file_path is not None:
            self._params.update(mx.nd.load(params_file_path))
        with open(symbol_json_file_path) as f:
            self._json = json.loads(mx.symbol.load_json(f.read()).tojson())

        ## build node graph
        self._start_node_name = self._json['nodes'][0]['name']
        self._node_map = defaultdict(lambda: defaultdict(dict))
        for node_id, node in enumerate(self._json['nodes']):
            self._node_map[node['name']] = node
            node['op'] = None if node['op'] == 'null' else node['op']
            node['is_aux'] = (node['name'].endswith('moving_mean') or node['name'].endswith('moving_var'))
            node['input_nodes'] = []
            for i in node['inputs']:
                child_node = self._json['nodes'][i[0]]
                node['input_nodes'].append(child_node)
                if 'input_nodes_of' not in child_node:
                    child_node['input_nodes_of'] = []
                child_node['input_nodes_of'].append(node)

        self._out_map = {}

        def _find_next(curr_node=None, curr_output=None, prev_output=None, fast_stop=False):
            if curr_node is None:
                return
            else:
                curr_output = profile.merge(curr_output, curr_node, self._params)
                if profile.ready(curr_output):
                    self._out_map[curr_output['_id']] = curr_output
                    for n in prev_output:
                        if curr_output['_id'] not in self._out_map[n['_id']]['next']:
                            self._out_map[n['_id']]['next'].append(curr_output['_id'])
                    prev_output = [curr_output, ]
                    if fast_stop:
                        return
                    if 'input_nodes_of' in curr_node:
                        for i,o in enumerate(curr_node['input_nodes_of']):
                            _find_next(curr_node=o, prev_output=prev_output, curr_output=profile.empty_output(),
                                       fast_stop=fast_stop or (profile.fast_stop and (i >= 1)))
                else:
                    if 'input_nodes_of' in curr_node:
                        curr_output_cp = copy(curr_output)
                        for i,o in enumerate(curr_node['input_nodes_of']):
                            nc = copy(curr_output_cp)
                            _find_next(curr_node=o, prev_output=prev_output, curr_output=nc,
                                       fast_stop=fast_stop or (profile.fast_stop and (i >= 1)))  # no copy here

        _find_next(curr_node=self._node_map[self._start_node_name],
                   curr_output=profile.empty_output(),
                   prev_output=[])
        profile.reset_id()
        for v in self._out_map.values():
            v['next'] = [self._out_map[i] for i in v['next']]

        self._queue = [profile.strip_output(n) for n in sorted(self._out_map.values(), key=lambda x: x["_id"])]
        self.length = len(self._queue)

    def __iter__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            for x in self._queue:
                yield x

    def __len__(self):
        return self.length

    def reset(self):
        self.current = 0
