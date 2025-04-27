import fcntl
import struct
import threading
from typing import Dict, List, Optional, Type, Union

import bacpypes

from pydantic import ConfigDict, Field
from typing_extensions import Literal

import sys
import ipaddress
from ipaddress import IPv4Network
import socket

from bacpypes.debugging import bacpypes_debugging, ModuleLogger
from bacpypes.consolelogging import ConfigArgumentParser
from bacpypes.consolecmd import ConsoleCmd

from bacpypes.core import run, deferred, enable_sleeping
from bacpypes.iocb import IOCB

from bacpypes.pdu import Address
from bacpypes.object import get_datatype

from bacpypes.apdu import SimpleAckPDU, \
    ReadPropertyRequest, ReadPropertyACK, WritePropertyRequest
from bacpypes.primitivedata import Null, Atomic, Boolean, Unsigned, Integer, \
    Real, Double, OctetString, CharacterString, BitString, Date, Time, ObjectIdentifier
from bacpypes.constructeddata import Array, Any, AnyAtomic
from bacpypes.iocb import IOCB, IOController
from bacpypes.pdu import Address, GlobalBroadcast
from bacpypes.apdu import WhoIsRequest, IAmRequest
#from bacpypes.core import run, deferred, enable_sleeping

from bacpypes.app import BIPSimpleApplication
from bacpypes.local.device import LocalDeviceObject

from inference.core.managers.base import ModelManager
from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    INTEGER_KIND,
    BOOLEAN_KIND,
    Selector,    
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode

LONG_DESCRIPTION = """
This **BACnet IP** block integrates a Roboflow Workflow with a BACnet device via UDP.
It can write to the present value property of listed BACnet object class types.

Note this block requires privilaged access to the network interface card. If inference
is running in a docker container, this can be provided using --network=host or ideally
by setting up a bridged VLAN (ex. Macvlan).
"""

# BACnet class names vs. class ids
BACNET_CLASSES = {
    "Analog Input":(0,lambda x: Real(float(x))),
    "Analog Output":(1,lambda x: Real(float(x))),
    "Analog Value":(2,lambda x: Real(float(x))),
    "Binary Input":(3,Boolean),
    "Binary Output":(4,Boolean),
    "Binary Value":(5,Boolean),
    "Multi-State Input":(13,lambda x: Integer(int(x))),
    "Multi-State Output":(14,lambda x: Integer(int(x))),
    "Multi-State Value":(19,lambda x: Integer(int(x))),
}

# start with present value, consider writes to other properties later
BACNET_WRITE_PROPERTY = 85

bacnet_class_names = list(BACNET_CLASSES.keys())

# bacnet address is 32 bits:
#   first 10 bits is the class id
#   next 22 bits is the object id
def to_bacoid(bacnet_class_id,bacnet_object_id):
    shifted_class = bacnet_class_id << 22
    return shifted_class | bacnet_object_id

def is_valid_ip(ip_string):
    if not ip_string:
        return False
    try:
        ipaddress.ip_address(ip_string)
        return True
    except ValueError:
        return False

net_masks = {}
def get_ip_and_subnet_info():
    global net_masks
    if net_masks:
        return net_masks
    net_masks = [(if_name,get_netmask(if_name),get_ip_address(if_name)) for _, if_name in socket.if_nameindex()]
    net_masks = {name:f"{ip}/{IPv4Network(f'0.0.0.0/{mask}').prefixlen}" for name, mask, ip in net_masks if mask and ip and is_valid_ip(ip)}
    return net_masks

def get_netmask(interface):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x891b, struct.pack('256s', interface.encode('utf-8')[:15]))[20:24])
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x891b, struct.pack('256s', interface.encode('utf-8')[:15]))[20:24])
    except OSError:        
        return None

def get_ip_address(interface):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', interface.encode('utf-8')[:15])
        )[20:24])
    except OSError:        
        return None

class BacnetIpManifest(WorkflowBlockManifest):

    model_config = ConfigDict(
        json_schema_extra={
            "name": "BACnet IP",
            "version": "v1",
            "short_description": "Generic BACnet IP write using bacpypes.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )

    type: Literal["roboflow_core/bacnet_ip@v1"]

    network_interface: Union[
        Literal[tuple(get_ip_and_subnet_info().keys())],
    ] = Field(
        default="eth0" if "eth0" in get_ip_and_subnet_info() else None if not get_ip_and_subnet_info() else get_ip_and_subnet_info().keys()[0],
        description="The network adapter to bind to",
    )
    value_to_write: Union[str] = Field(
        description="Value to write to BACnet object's present value"
    )
    inference_device_id: Union[int] = Field(
        description="Device ID for the workflow server's device.", default=78910
    )
    bacnet_port: int = Field(
        default=47808,
        description="Port number for BACnet communication.",
        examples=[47808],
    )
    device_id_to_write: Union[str,Selector(kind=[INTEGER_KIND])] = Field(
        description="Device ID or IP address to write to.",
        examples=["1234","1.1.1.1"]
    )
    class_to_write: Union[
        Selector(kind=[STRING_KIND]),
        Literal[tuple(bacnet_class_names)],
    ] = Field(
        default=bacnet_class_names[0],
        description="The BACnet object class type to write to.",
        examples=[k for k in bacnet_class_names]
    )
    object_id_to_write: Union[int] = Field(
        description="Object ID to write to.", examples=["0"]
    )
    priority: Union[int] = Field(
        description="Write priority.", examples=["12"], default=12
    )
    depends_on: Selector() = Field(
        description="Reference to the step output this block depends on.",
        examples=["$steps.some_previous_step"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
        examples=[True, "$inputs.fire_and_forget"],
    )
    response_timeout: int = Field(
        default=5,
        description="Number of seconds to wait for a response if fire and forget is false.",
        examples=[5],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        get_ip_and_subnet_info()
        return [OutputDefinition(name="write_reply", kind=[LIST_OF_VALUES_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

# this is shared between all bacnet blocks
bacnet_app = None

class BacnetIpV1(WorkflowBlock):

    """A BACnet IP communication block using bacpypes.
    """

    iocb_responded = threading.Event()

    class BacnetApplication(BIPSimpleApplication):
        def __init__(self, *args):
            BIPSimpleApplication.__init__(self, *args)

            # keep track of requests to line up responses
            self._request = None

        def request(self, apdu):
            # save a copy of the request
            if isinstance(apdu, WhoIsRequest):
                self._request = apdu

            # forward it along
            BIPSimpleApplication.request(self, apdu)

        def indication(self, apdu):
            print("response: {}", apdu)  
            # response?
            BIPSimpleApplication.indication(self, apdu)

        def confirmation(self, apdu):
            print("confirmation: {}", apdu)  
            # forward it along
            BIPSimpleApplication.confirmation(self, apdu)

    # static wrapper class
    class BacnetAppWrapper:

        app_thread = None        
        inference_device_id = None
        bacnet_port = None
        network_interface = None
        inference_device = None

        @classmethod
        def get_app(cls,inference_device_id,bacnet_port,network_interface):
            global bacnet_app

            if bacnet_app:
                if inference_device_id!=bacnet_app.inference_device_id or bacnet_port!=bacnet_app.bacnet_port or network_interface!=bacnet_app.network_interface:
                    try:
                        bacpypes.core.stop()
                    finally:
                        pass
                    bacnet_app = BacnetIpV1.BacnetAppWrapper(inference_device_id,bacnet_port,network_interface)
                    return bacnet_app
                else:
                    return bacnet_app
            
            bacnet_app = BacnetIpV1.BacnetAppWrapper(inference_device_id,bacnet_port,network_interface)
            return bacnet_app


        def __init__(self,inference_device_id,bacnet_port,network_interface):
            
            global bacnet_app

            self.ip_bindings = []
            self.app = None

            self.inference_device_id = inference_device_id
            self.bacnet_port = bacnet_port
            self.network_interface = network_interface                        
            self.app_created = threading.Event()           
            self.app_thread = threading.Thread(target=self.start_app, args=())
            self.app_thread.daemon = True
            self.app_thread.start()
            #self.start_app(inference_device_id,bacnet_port)

        # starts bacpypes device and app once
        def start_app(self):                        
            print("starting bacnet app")
            self.inference_device = LocalDeviceObject(
                objectIdentifier=self.inference_device_id,
                objectName='Inference',
                vendorIdentifier=9999,
                segmentationSupported=True
            )                
            nics = get_ip_and_subnet_info()
            address_str = f"{nics[self.network_interface]}:{self.bacnet_port}"
            #address_str = f"{nics[self.network_interface]}"                
            self.app = BacnetIpV1.BacnetApplication(self.inference_device, Address(address_str))
            print(f"BACnet app started and bound to address {address_str} for {self.network_interface}")
            self.who_is()
            self.app_created.set()
            bacpypes.core.run()

        def wait_for_app(self):
            self.app_created.wait()

        def app(self):
            return self.app

        def who_is(self):
            
            print(f"whois {self.app.who_is(None, None, GlobalBroadcast())}")
            #print(f"whois {self.app.who_is(None, None, Address('172.17.255.255'))}")
            #self.app.i_am()

        def request_io(self,request):
            print("request")
            return self.app.request_io(request)

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["step_execution_mode"]

    def __init__(
        self,
        step_execution_mode: StepExecutionMode,
    ):
        self.bacnet_app = None
        self._step_execution_mode = step_execution_mode

    def __del__(self):
        pass

    def write_value(self,device_id,write_bacoid,value,priority,fire_and_forget,response_timeout):
        
        # make sure the app has started
        self.bacnet_app.wait_for_app()

        # build a request
        request = WritePropertyRequest(
            objectIdentifier=write_bacoid,            
            propertyIdentifier=BACNET_WRITE_PROPERTY
            )
        # if it's a device id we have to reference bindings
        # if it's an ip address, just set the destination
        if is_valid_ip(device_id):
            request.pduDestination = Address(device_id)
            print(f"destination: {request.pduDestination}")
        else:
            if device_id in self._ip_bindings:
                #request.pduDestination = Address(self._ip_bindings[device_id])
                request.pduDestination = device_id
            else:
                raise ValueError(
                    f"BACnet device id binding unknown: {device_id}"
                )
            
        # save the value
        request.propertyValue = Any()
        try:            
            request.propertyValue.cast_in(value) # TODO: we might need the right type here
            request.priority = priority
            print(f"request: {request} value: {request.propertyValue}")
        except Exception as error: 
            raise ValueError(
                f"Write property case error: {error}"
            )

        iocb = IOCB(request)

        # response callback        
        iocb.add_callback(self.on_response)

        self.iocb_responded.clear()
        deferred(self.bacnet_app.request_io, iocb)

        if not fire_and_forget:
            self.iocb_responded.wait(timeout=response_timeout)
        
        return iocb

    def on_response(self, iocb):
        self.iocb_responded.set()
        print(f"iocb {iocb.ioComplete.is_set()} {iocb.ioState}")

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BacnetIpManifest

    def run(
        self,
        network_interface: str,
        value_to_write: str,
        inference_device_id: int,
        bacnet_port: int,
        device_id_to_write: str,
        class_to_write: str,
        object_id_to_write: int,
        fire_and_forget: bool,
        depends_on: any,
        priority: int,
        response_timeout: int
    ) -> dict:

        self.bacnet_app = BacnetIpV1.BacnetAppWrapper.get_app(inference_device_id,bacnet_port,network_interface)

        if self._step_execution_mode == StepExecutionMode.LOCAL:

            # get the class id and data type for the class we want to write to
            (class_id,data_type) = BACNET_CLASSES[class_to_write]
            
            # convert the value string into the datatype we want to write to
            encodable_value = data_type(value_to_write)

            # bacoid to write to
            write_bacoid = to_bacoid(class_id,object_id_to_write)
            

            # write the value and wait for a response
            iocb = self.write_value(device_id_to_write,write_bacoid,encodable_value,priority,fire_and_forget,response_timeout)

            # return the response
            
            if not isinstance(iocb.ioResponse, SimpleAckPDU):
                return {"write_reply": ["write response is not an ack"]}   
            if iocb.ioError:
                print(f"iocb_err:{str(iocb.ioError)}")
                return {"write_reply": [str(iocb.ioError)]}    
            else:
                print(f"iocb_err:{str(iocb.ioResponse)}")
                return {"write_reply": [str(iocb.ioResponse)]}
            
            #return {"write_reply": [str(f"{iocb}")]}
            #return {"write_reply": [None]}

        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Depth Estimation. Please use a local or dedicated inference server."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )