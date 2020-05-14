#!/usr/bin/env python3


import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None, plugin=None):
                
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Plugin initialization for specified device
        # and load extensions library if specified
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device = device)
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

        # Read IR
        log.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR to the plugin...")
        
        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = \
                [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by "
                          "the plugin for specified device {}:\n {}".
                          format(self.plugin.device,
                                 ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)
        
        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.net_plugin = self.plugin.load(network=self.net)
        else:
            self.net_plugin = self.plugin.load(network=self.net, num_requests=num_requests)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        assert len(self.net.inputs.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.net.inputs))
        assert len(self.net.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.net.outputs))
        
        ### Note: You may need to update the function parameters. ###
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        
        self.infer_request_handle = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        
        return self.net_plugin
        ### Note: You may need to update the function parameters. ###

    def wait(self, request_id):
        
        wait_process = self.net_plugin.requests[request_id].wait(-1)
                
        return wait_process
        ### Note: You may need to update the function parameters. ###

    def get_output(self, request_id, output=None):
        
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res
    
        ### Note: You may need to update the function parameters. ###
        
    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer  
        """
        perf_count = self.net_plugin.requests[request_id].get_perf_counts()
        return perf_count