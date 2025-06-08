import re
import socket
import threading
import numpy as np
import time

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

ONVIF_WORKFLOW = {
  "version": "1.0",
  "inputs": [
    {
      "type": "InferenceImage",
      "name": "image"
    }
  ],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "model_1",
      "images": "$inputs.image",
      "model_id": "yolov10n-640"
    },
    {
      "type": "roboflow_core/onvif_sink@v1",
      "name": "onvif_control",
      "predictions": "$steps.byte_tracker.tracked_detections",
      "camera_ip": "localhost",
      "camera_username": "admin",
      "camera_password": "123456",
      "default_position_preset": "1",
      "pid_ki": 0,
      "pid_kp": 0.25,
      "pid_kd": 1,
      "zoom_if_able": False,
      "move_to_position_after_idle_seconds": 0,
      "minimum_camera_speed": 10,
      "dead_zone": 100,
      "movement_type": "Follow",
      "camera_update_rate_limit": 500,
      "camera_port": 1981
    },
    {
      "type": "roboflow_core/detections_filter@v1",
      "name": "detections_filter",
      "predictions": "$steps.model_1.predictions",
      "operations": [
        {
          "type": "DetectionsFilter",
          "filter_operation": {
            "type": "StatementGroup",
            "operator": "and",
            "statements": [
              {
                "type": "BinaryStatement",
                "negate": False,
                "left_operand": {
                  "type": "DynamicOperand",
                  "operand_name": "_",
                  "operations": [
                    {
                      "type": "ExtractDetectionProperty",
                      "property_name": "class_name"
                    }
                  ]
                },
                "comparator": {
                  "type": "in (Sequence)"
                },
                "right_operand": {
                  "type": "StaticOperand",
                  "value": [
                    "banana"
                  ]
                }
              }
            ]
          }
        }
      ],
      "operations_parameters": {},
      "uistate": {
        "selectedFilterType": "filter_by_class_and_confidence",
        "isClassFilteringActive": True,
        "isConfidenceFilteringActive": False,
        "classSetInclusionMode": "in",
        "classList": [
          "banana"
        ],
        "confidenceThreshold": 0.5,
        "confidenceOperator": ">=",
        "referenceImage": "$inputs.image",
        "sizeThreshold": 5,
        "sizeThresholdOperator": "<=",
        "zoneOperator": "in",
        "isZoneStatic": True,
        "zonePoints": [],
        "detectionReferencePoint": "center",
        "dynamicZone": None,
        "detectionsOffset": [
          0,
          0
        ],
        "parentClassName": None
      }
    },
    {
      "type": "roboflow_core/byte_tracker@v3",
      "name": "byte_tracker",
      "image": "$inputs.image",
      "detections": "$steps.detections_filter.predictions"
    }
  ],
  "outputs": [
    {
      "type": "JsonField",
      "name": "output",
      "coordinates_system": "own",
      "selector": "$steps.onvif_control.*"
    }
  ]
}

# responses are from https://www.onvif.org/wp-content/uploads/2016/12/ONVIF_WG-APG-Application_Programmers_Guide-1.pdf
ONVIF_SOAP_RESPONSES = {
    "http://www.onvif.org/ver10/device/wsdl/GetCapabilities": '''
<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
	xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
	xmlns:SOAP-ENC="http://www.w3.org/2003/05/soap-encoding"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xmlns:xsd="http://www.w3.org/2001/XMLSchema"
	xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
	xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
	xmlns:chan="http://schemas.microsoft.com/ws/2005/02/duplex"
	xmlns:wsa5="http://www.w3.org/2005/08/addressing"
	xmlns:xmime="http://www.w3.org/2005/05/xmlmime"
	xmlns:xop="http://www.w3.org/2004/08/xop/include"
	xmlns:wsrfbf="http://docs.oasis-open.org/wsrf/bf-2"
	xmlns:tt="http://www.onvif.org/ver10/schema"
	xmlns:wstop="http://docs.oasis-open.org/wsn/t-1"
	xmlns:wsrfr="http://docs.oasis-open.org/wsrf/r-2"
	xmlns:tan="http://www.onvif.org/ver20/analytics/wsdl"
	xmlns:tdn="http://www.onvif.org/ver10/network/wsdl"
	xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
	xmlns:tev="http://www.onvif.org/ver10/events/wsdl"
	xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
	xmlns:c14n="http://www.w3.org/2001/10/xml-exc-c14n#"
	xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
	xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
	xmlns:wsc="http://schemas.xmlsoap.org/ws/2005/02/sc"
	xmlns:ds="http://www.w3.org/2000/09/xmldsig#"
	xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
	xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl"
	xmlns:tmd="http://www.onvif.org/ver10/deviceIO/wsdl"
	xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
	xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
	xmlns:ter="http://www.onvif.org/ver10/error"
	xmlns:tns1="http://www.onvif.org/ver10/topics"
	xmlns:trt2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tr2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tplt="http://www.onvif.org/ver10/plus/schema"
	xmlns:tpl="http://www.onvif.org/ver10/plus/wsdl"
	xmlns:ewsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:exsd="http://www.onvifext.com/onvif/ext/ver10/schema"
	xmlns:tnshik="http://www.hikvision.com/2011/event/topics"
	xmlns:hikwsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:hikxsd="http://www.onvifext.com/onvif/ext/ver10/schema">
	<SOAP-ENV:Header>
		<wsa5:MessageID>urn:uuid:d23b6ce5-1d3b-43b5-8078-5c1b3f4a7377</wsa5:MessageID>
		<wsa5:To SOAP-ENV:mustUnderstand="true">http://127.0.0.1:1981/onvif/device_service</wsa5:To>
		<wsa5:Action SOAP-ENV:mustUnderstand="true">http://www.onvif.org/ver10/device/wsdl/GetCapabilities</wsa5:Action>
		<wsse:Security>
			<wsse:UsernameToken>
				<wsse:Username>admin</wsse:Username>
				<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">VVV9UyaHXYIjreK3BeGZDPsILr8=</wsse:Password>
				<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">cy3CX8qvSO9SBHdz20AYcg==</wsse:Nonce>
				<wsu:Created>2025-06-08T20:50:59+00:00</wsu:Created>
			</wsse:UsernameToken>
		</wsse:Security>
	</SOAP-ENV:Header>
	<SOAP-ENV:Body>
		<tds:GetCapabilitiesResponse>
			<tds:Capabilities>
				<tt:Analytics>
					<tt:XAddr>http://127.0.0.1:1981/onvif/analytics</tt:XAddr>
					<tt:RuleSupport>true</tt:RuleSupport>
					<tt:AnalyticsModuleSupport>true</tt:AnalyticsModuleSupport>
				</tt:Analytics>
				<tt:Device>
					<tt:XAddr>http://127.0.0.1:1981/onvif/device</tt:XAddr>
					<tt:Network>
						<tt:IPFilter>false</tt:IPFilter>
						<tt:ZeroConfiguration>false</tt:ZeroConfiguration>
						<tt:IPVersion6>false</tt:IPVersion6>
						<tt:DynDNS>false</tt:DynDNS>
						<tt:Extension>
							<tt:Dot11Configuration>false</tt:Dot11Configuration>
						</tt:Extension>
					</tt:Network>
					<tt:System>
						<tt:DiscoveryResolve>false</tt:DiscoveryResolve>
						<tt:DiscoveryBye>false</tt:DiscoveryBye>
						<tt:RemoteDiscovery>false</tt:RemoteDiscovery>
						<tt:SystemBackup>false</tt:SystemBackup>
						<tt:SystemLogging>true</tt:SystemLogging>
						<tt:FirmwareUpgrade>true</tt:FirmwareUpgrade>
						<tt:SupportedVersions>
							<tt:Major>17</tt:Major>
							<tt:Minor>6</tt:Minor>
						</tt:SupportedVersions>
						<tt:Extension>
							<tt:HttpFirmwareUpgrade>true</tt:HttpFirmwareUpgrade>
							<tt:HttpSystemBackup>false</tt:HttpSystemBackup>
							<tt:HttpSystemLogging>true</tt:HttpSystemLogging>
							<tt:HttpSupportInformation>true</tt:HttpSupportInformation>
						</tt:Extension>
					</tt:System>
					<tt:IO>
						<tt:InputConnectors>1</tt:InputConnectors>
						<tt:RelayOutputs>1</tt:RelayOutputs>
					</tt:IO>
					<tt:Security>
						<tt:TLS1.1>false</tt:TLS1.1>
						<tt:TLS1.2>true</tt:TLS1.2>
						<tt:OnboardKeyGeneration>false</tt:OnboardKeyGeneration>
						<tt:AccessPolicyConfig>false</tt:AccessPolicyConfig>
						<tt:X.509Token>false</tt:X.509Token>
						<tt:SAMLToken>false</tt:SAMLToken>
						<tt:KerberosToken>false</tt:KerberosToken>
						<tt:RELToken>false</tt:RELToken>
					</tt:Security>
				</tt:Device>
				<tt:Events>
					<tt:XAddr>http://127.0.0.1:1981/onvif/events</tt:XAddr>
					<tt:WSSubscriptionPolicySupport>true</tt:WSSubscriptionPolicySupport>
					<tt:WSPullPointSupport>true</tt:WSPullPointSupport>
					<tt:WSPausableSubscriptionManagerInterfaceSupport>true</tt:WSPausableSubscriptionManagerInterfaceSupport>
				</tt:Events>
				<tt:Imaging>
					<tt:XAddr>http://127.0.0.1:1981/onvif/imaging</tt:XAddr>
				</tt:Imaging>
				<tt:Media>
					<tt:XAddr>http://127.0.0.1:1981/onvif/media</tt:XAddr>
					<tt:StreamingCapabilities>
						<tt:RTPMulticast>true</tt:RTPMulticast>
						<tt:RTP_TCP>true</tt:RTP_TCP>
						<tt:RTP_RTSP_TCP>true</tt:RTP_RTSP_TCP>
					</tt:StreamingCapabilities>
				</tt:Media>
				<tt:PTZ>
					<tt:XAddr>http://127.0.0.1:1981/onvif/ptz</tt:XAddr>
				</tt:PTZ>
				<tt:Extension>
					<hikxsd:hikCapabilities>
						<hikxsd:XAddr>http://127.0.0.1:1981/onvif/hik_ext</hikxsd:XAddr>
						<hikxsd:IOInputSupport>true</hikxsd:IOInputSupport>
					</hikxsd:hikCapabilities>
					<tt:DeviceIO>
						<tt:XAddr>http://127.0.0.1:1981/onvif/deviceIO</tt:XAddr>
						<tt:VideoSources>1</tt:VideoSources>
						<tt:VideoOutputs>0</tt:VideoOutputs>
						<tt:AudioSources>1</tt:AudioSources>
						<tt:AudioOutputs>1</tt:AudioOutputs>
						<tt:RelayOutputs>1</tt:RelayOutputs>
					</tt:DeviceIO>
					<tt:Extensions>
						<tt:TelexCapabilities>
							<tt:XAddr>http://127.0.0.1:1981/onvif/telecom_service</tt:XAddr>
							<tt:TimeOSDSupport>true</tt:TimeOSDSupport>
							<tt:TitleOSDSupport>true</tt:TitleOSDSupport>
							<tt:PTZ3DZoomSupport>true</tt:PTZ3DZoomSupport>
							<tt:PTZAuxSwitchSupport>true</tt:PTZAuxSwitchSupport>
							<tt:MotionDetectorSupport>true</tt:MotionDetectorSupport>
							<tt:TamperDetectorSupport>true</tt:TamperDetectorSupport>
						</tt:TelexCapabilities>
					</tt:Extensions>
					<ewsd:hbCapabilities>
						<exsd:XAddr>http://127.0.0.1:1981/onvif/hbgk_ext</exsd:XAddr>
						<exsd:H265Support>true</exsd:H265Support>
						<exsd:PrivacyMaskSupport>true</exsd:PrivacyMaskSupport>
						<exsd:CameraNum>1</exsd:CameraNum>
						<exsd:MaxMaskAreaNum>4</exsd:MaxMaskAreaNum>
					</ewsd:hbCapabilities>
					<tplt:Plus>
						<tplt:XAddr>http://127.0.0.1:1981/onvif/hbgk_ext</tplt:XAddr>
						<tplt:H265>true</tplt:H265>
						<tplt:PrivacyMask>true</tplt:PrivacyMask>
						<tplt:CameraNum>1</tplt:CameraNum>
						<tplt:MaxMaskAreaNum>4</tplt:MaxMaskAreaNum>
					</tplt:Plus>
				</tt:Extension>
			</tds:Capabilities>
		</tds:GetCapabilitiesResponse>
	</SOAP-ENV:Body>
</SOAP-ENV:Envelope>
''',
"http://www.onvif.org/ver10/media/wsdl/GetProfiles":'''
<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
	xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
	xmlns:SOAP-ENC="http://www.w3.org/2003/05/soap-encoding"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xmlns:xsd="http://www.w3.org/2001/XMLSchema"
	xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
	xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
	xmlns:chan="http://schemas.microsoft.com/ws/2005/02/duplex"
	xmlns:wsa5="http://www.w3.org/2005/08/addressing"
	xmlns:xmime="http://www.w3.org/2005/05/xmlmime"
	xmlns:xop="http://www.w3.org/2004/08/xop/include"
	xmlns:wsrfbf="http://docs.oasis-open.org/wsrf/bf-2"
	xmlns:tt="http://www.onvif.org/ver10/schema"
	xmlns:wstop="http://docs.oasis-open.org/wsn/t-1"
	xmlns:wsrfr="http://docs.oasis-open.org/wsrf/r-2"
	xmlns:tan="http://www.onvif.org/ver20/analytics/wsdl"
	xmlns:tdn="http://www.onvif.org/ver10/network/wsdl"
	xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
	xmlns:tev="http://www.onvif.org/ver10/events/wsdl"
	xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
	xmlns:c14n="http://www.w3.org/2001/10/xml-exc-c14n#"
	xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
	xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
	xmlns:wsc="http://schemas.xmlsoap.org/ws/2005/02/sc"
	xmlns:ds="http://www.w3.org/2000/09/xmldsig#"
	xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
	xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl"
	xmlns:tmd="http://www.onvif.org/ver10/deviceIO/wsdl"
	xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
	xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
	xmlns:ter="http://www.onvif.org/ver10/error"
	xmlns:tns1="http://www.onvif.org/ver10/topics"
	xmlns:trt2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tr2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tplt="http://www.onvif.org/ver10/plus/schema"
	xmlns:tpl="http://www.onvif.org/ver10/plus/wsdl"
	xmlns:ewsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:exsd="http://www.onvifext.com/onvif/ext/ver10/schema"
	xmlns:tnshik="http://www.hikvision.com/2011/event/topics"
	xmlns:hikwsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:hikxsd="http://www.onvifext.com/onvif/ext/ver10/schema">
	<SOAP-ENV:Header>
		<wsa5:MessageID>urn:uuid:5bd34f66-b7e3-45b8-a39c-7bbab266ca44</wsa5:MessageID>
		<wsa5:To SOAP-ENV:mustUnderstand="true">http://192.168.1.253:80/onvif/media</wsa5:To>
		<wsa5:Action SOAP-ENV:mustUnderstand="true">http://www.onvif.org/ver10/media/wsdl/GetProfiles</wsa5:Action>
		<wsse:Security>
			<wsse:UsernameToken>
				<wsse:Username>admin</wsse:Username>
				<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">rMOYRxmtFwxAAE9kVh6/VQOXijU=</wsse:Password>
				<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">YCo3BVqFNhU5xOqEBI2cRw==</wsse:Nonce>
				<wsu:Created>2025-06-08T21:55:38+00:00</wsu:Created>
			</wsse:UsernameToken>
		</wsse:Security>
	</SOAP-ENV:Header>
	<SOAP-ENV:Body>
		<trt:GetProfilesResponse>
			<trt:Profiles token="MainStream" fixed="false">
				<tt:Name>MainStream</tt:Name>
				<tt:VideoSourceConfiguration token="VideoSourceMain">
					<tt:Name>VideoSourceMain</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:SourceToken>VideoSourceMain</tt:SourceToken>
					<tt:Bounds x="0" y="0" width="3072" height="2048"></tt:Bounds>
				</tt:VideoSourceConfiguration>
				<tt:AudioSourceConfiguration token="AudioMainToken">
					<tt:Name>AudioMainName</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:SourceToken>AudioMainSrcToken</tt:SourceToken>
				</tt:AudioSourceConfiguration>
				<tt:VideoEncoderConfiguration token="VideoEncodeMain">
					<tt:Name>VideoEncodeMain</tt:Name>
					<tt:UseCount>1</tt:UseCount>
					<tt:Encoding>H264</tt:Encoding>
					<tt:Resolution>
						<tt:Width>3072</tt:Width>
						<tt:Height>2048</tt:Height>
					</tt:Resolution>
					<tt:Quality>50</tt:Quality>
					<tt:RateControl>
						<tt:FrameRateLimit>25</tt:FrameRateLimit>
						<tt:EncodingInterval>1</tt:EncodingInterval>
						<tt:BitrateLimit>4000</tt:BitrateLimit>
					</tt:RateControl>
					<tt:MPEG4>
						<tt:GovLength>0</tt:GovLength>
						<tt:Mpeg4Profile>SP</tt:Mpeg4Profile>
					</tt:MPEG4>
					<tt:H264>
						<tt:GovLength>100</tt:GovLength>
						<tt:H264Profile>High</tt:H264Profile>
					</tt:H264>
					<tt:Multicast>
						<tt:Address>
							<tt:Type>IPv4</tt:Type>
							<tt:IPv4Address>192.168.1.253</tt:IPv4Address>
						</tt:Address>
						<tt:Port>0</tt:Port>
						<tt:TTL>0</tt:TTL>
						<tt:AutoStart>false</tt:AutoStart>
					</tt:Multicast>
					<tt:SessionTimeout>PT00H12M00S</tt:SessionTimeout>
				</tt:VideoEncoderConfiguration>
				<tt:AudioEncoderConfiguration token="G711">
					<tt:Name>AudioMain</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:Encoding>G711</tt:Encoding>
					<tt:Bitrate>64000</tt:Bitrate>
					<tt:SampleRate>8000</tt:SampleRate>
					<tt:Multicast>
						<tt:Address>
							<tt:Type>IPv4</tt:Type>
							<tt:IPv4Address>192.168.1.253</tt:IPv4Address>
						</tt:Address>
						<tt:Port>80</tt:Port>
						<tt:TTL>1</tt:TTL>
						<tt:AutoStart>false</tt:AutoStart>
					</tt:Multicast>
					<tt:SessionTimeout>PT00H00M00.060S</tt:SessionTimeout>
				</tt:AudioEncoderConfiguration>
				<tt:VideoAnalyticsConfiguration token="VideoAnalyticsToken">
					<tt:Name>VideoAnalyticsName</tt:Name>
					<tt:UseCount>3</tt:UseCount>
					<tt:AnalyticsEngineConfiguration>
						<tt:AnalyticsModule Type="tt:CellMotionEngine" Name="MyCellMotionModule">
							<tt:Parameters>
								<tt:SimpleItem Name="Sensitivity" Value="80"></tt:SimpleItem>
								<tt:ElementItem Name="Layout">
									<tt:CellLayout Rows="18" Columns="22">
										<tt:Transformation>
											<tt:Translate y="-1" x="-1" />
											<tt:Scale y="9.99999997E-07" x="9.99999997E-07" />
										</tt:Transformation>
									</tt:CellLayout>
								</tt:ElementItem>
							</tt:Parameters>
						</tt:AnalyticsModule>
					</tt:AnalyticsEngineConfiguration>
					<tt:RuleEngineConfiguration>
						<tt:Rule Type="tt:CellMotionDetector" Name="MyMotionDetectorRule">
							<tt:Parameters>
								<tt:SimpleItem Name="MinCount" Value="5"></tt:SimpleItem>
								<tt:SimpleItem Name="AlarmOnDelay" Value="100"></tt:SimpleItem>
								<tt:SimpleItem Name="AlarmOffDelay" Value="100"></tt:SimpleItem>
								<tt:SimpleItem Name="ActiveCells" Value="0P8A8A=="></tt:SimpleItem>
							</tt:Parameters>
						</tt:Rule>
					</tt:RuleEngineConfiguration>
				</tt:VideoAnalyticsConfiguration>
				<tt:PTZConfiguration token="ptz0">
					<tt:Name>ptz0</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:NodeToken>ptz0</tt:NodeToken>
					<tt:DefaultAbsolutePantTiltPositionSpace>http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace</tt:DefaultAbsolutePantTiltPositionSpace>
					<tt:DefaultAbsoluteZoomPositionSpace>http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace</tt:DefaultAbsoluteZoomPositionSpace>
					<tt:DefaultRelativePanTiltTranslationSpace>http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationGenericSpace</tt:DefaultRelativePanTiltTranslationSpace>
					<tt:DefaultRelativeZoomTranslationSpace>http://www.onvif.org/ver10/tptz/ZoomSpaces/TranslationGenericSpace</tt:DefaultRelativeZoomTranslationSpace>
					<tt:DefaultContinuousPanTiltVelocitySpace>http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace</tt:DefaultContinuousPanTiltVelocitySpace>
					<tt:DefaultContinuousZoomVelocitySpace>http://www.onvif.org/ver10/tptz/ZoomSpaces/VelocityGenericSpace</tt:DefaultContinuousZoomVelocitySpace>
					<tt:DefaultPTZSpeed>
						<tt:PanTilt x="1" y="1" space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/GenericSpeedSpace"></tt:PanTilt>
						<tt:Zoom x="1" space="http://www.onvif.org/ver10/tptz/ZoomSpaces/ZoomGenericSpeedSpace"></tt:Zoom>
					</tt:DefaultPTZSpeed>
					<tt:DefaultPTZTimeout>PT00H01M00S</tt:DefaultPTZTimeout>
					<tt:PanTiltLimits>
						<tt:Range>
							<tt:URI>http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace</tt:URI>
							<tt:XRange>
								<tt:Min>-1</tt:Min>
								<tt:Max>1</tt:Max>
							</tt:XRange>
							<tt:YRange>
								<tt:Min>-1</tt:Min>
								<tt:Max>1</tt:Max>
							</tt:YRange>
						</tt:Range>
					</tt:PanTiltLimits>
					<tt:ZoomLimits>
						<tt:Range>
							<tt:URI>http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace</tt:URI>
							<tt:XRange>
								<tt:Min>-1</tt:Min>
								<tt:Max>1</tt:Max>
							</tt:XRange>
						</tt:Range>
					</tt:ZoomLimits>
				</tt:PTZConfiguration>
			</trt:Profiles>
			<trt:Profiles token="SubStream" fixed="false">
				<tt:Name>SubStream</tt:Name>
				<tt:VideoSourceConfiguration token="VideoSourceMain">
					<tt:Name>VideoSourceMain</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:SourceToken>VideoSourceMain</tt:SourceToken>
					<tt:Bounds x="0" y="0" width="704" height="576"></tt:Bounds>
				</tt:VideoSourceConfiguration>
				<tt:AudioSourceConfiguration token="AudioMainToken">
					<tt:Name>AudioMainName</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:SourceToken>AudioMainSrcToken</tt:SourceToken>
				</tt:AudioSourceConfiguration>
				<tt:VideoEncoderConfiguration token="VideoEncodeSub">
					<tt:Name>VideoEncodeSub</tt:Name>
					<tt:UseCount>1</tt:UseCount>
					<tt:Encoding>H264</tt:Encoding>
					<tt:Resolution>
						<tt:Width>704</tt:Width>
						<tt:Height>576</tt:Height>
					</tt:Resolution>
					<tt:Quality>50</tt:Quality>
					<tt:RateControl>
						<tt:FrameRateLimit>25</tt:FrameRateLimit>
						<tt:EncodingInterval>1</tt:EncodingInterval>
						<tt:BitrateLimit>700</tt:BitrateLimit>
					</tt:RateControl>
					<tt:MPEG4>
						<tt:GovLength>0</tt:GovLength>
						<tt:Mpeg4Profile>SP</tt:Mpeg4Profile>
					</tt:MPEG4>
					<tt:H264>
						<tt:GovLength>100</tt:GovLength>
						<tt:H264Profile>High</tt:H264Profile>
					</tt:H264>
					<tt:Multicast>
						<tt:Address>
							<tt:Type>IPv4</tt:Type>
							<tt:IPv4Address>192.168.1.253</tt:IPv4Address>
						</tt:Address>
						<tt:Port>0</tt:Port>
						<tt:TTL>0</tt:TTL>
						<tt:AutoStart>false</tt:AutoStart>
					</tt:Multicast>
					<tt:SessionTimeout>PT00H12M00S</tt:SessionTimeout>
				</tt:VideoEncoderConfiguration>
				<tt:AudioEncoderConfiguration token="G711">
					<tt:Name>AudioMain</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:Encoding>G711</tt:Encoding>
					<tt:Bitrate>64000</tt:Bitrate>
					<tt:SampleRate>8000</tt:SampleRate>
					<tt:Multicast>
						<tt:Address>
							<tt:Type>IPv4</tt:Type>
							<tt:IPv4Address>192.168.1.253</tt:IPv4Address>
						</tt:Address>
						<tt:Port>80</tt:Port>
						<tt:TTL>1</tt:TTL>
						<tt:AutoStart>false</tt:AutoStart>
					</tt:Multicast>
					<tt:SessionTimeout>PT00H00M00.060S</tt:SessionTimeout>
				</tt:AudioEncoderConfiguration>
				<tt:VideoAnalyticsConfiguration token="VideoAnalyticsToken">
					<tt:Name>VideoAnalyticsName</tt:Name>
					<tt:UseCount>3</tt:UseCount>
					<tt:AnalyticsEngineConfiguration>
						<tt:AnalyticsModule Type="tt:CellMotionEngine" Name="MyCellMotionModule">
							<tt:Parameters>
								<tt:SimpleItem Name="Sensitivity" Value="80"></tt:SimpleItem>
								<tt:ElementItem Name="Layout">
									<tt:CellLayout Rows="18" Columns="22">
										<tt:Transformation>
											<tt:Translate y="-1" x="-1" />
											<tt:Scale y="9.99999997E-07" x="9.99999997E-07" />
										</tt:Transformation>
									</tt:CellLayout>
								</tt:ElementItem>
							</tt:Parameters>
						</tt:AnalyticsModule>
					</tt:AnalyticsEngineConfiguration>
					<tt:RuleEngineConfiguration>
						<tt:Rule Type="tt:CellMotionDetector" Name="MyMotionDetectorRule">
							<tt:Parameters>
								<tt:SimpleItem Name="MinCount" Value="5"></tt:SimpleItem>
								<tt:SimpleItem Name="AlarmOnDelay" Value="100"></tt:SimpleItem>
								<tt:SimpleItem Name="AlarmOffDelay" Value="100"></tt:SimpleItem>
								<tt:SimpleItem Name="ActiveCells" Value="0P8A8A=="></tt:SimpleItem>
							</tt:Parameters>
						</tt:Rule>
					</tt:RuleEngineConfiguration>
				</tt:VideoAnalyticsConfiguration>
				<tt:PTZConfiguration token="ptz0">
					<tt:Name>ptz0</tt:Name>
					<tt:UseCount>2</tt:UseCount>
					<tt:NodeToken>ptz0</tt:NodeToken>
					<tt:DefaultAbsolutePantTiltPositionSpace>http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace</tt:DefaultAbsolutePantTiltPositionSpace>
					<tt:DefaultAbsoluteZoomPositionSpace>http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace</tt:DefaultAbsoluteZoomPositionSpace>
					<tt:DefaultRelativePanTiltTranslationSpace>http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationGenericSpace</tt:DefaultRelativePanTiltTranslationSpace>
					<tt:DefaultRelativeZoomTranslationSpace>http://www.onvif.org/ver10/tptz/ZoomSpaces/TranslationGenericSpace</tt:DefaultRelativeZoomTranslationSpace>
					<tt:DefaultContinuousPanTiltVelocitySpace>http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace</tt:DefaultContinuousPanTiltVelocitySpace>
					<tt:DefaultContinuousZoomVelocitySpace>http://www.onvif.org/ver10/tptz/ZoomSpaces/VelocityGenericSpace</tt:DefaultContinuousZoomVelocitySpace>
					<tt:DefaultPTZSpeed>
						<tt:PanTilt x="1" y="1" space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/GenericSpeedSpace"></tt:PanTilt>
						<tt:Zoom x="1" space="http://www.onvif.org/ver10/tptz/ZoomSpaces/ZoomGenericSpeedSpace"></tt:Zoom>
					</tt:DefaultPTZSpeed>
					<tt:DefaultPTZTimeout>PT00H01M00S</tt:DefaultPTZTimeout>
					<tt:PanTiltLimits>
						<tt:Range>
							<tt:URI>http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace</tt:URI>
							<tt:XRange>
								<tt:Min>-1</tt:Min>
								<tt:Max>1</tt:Max>
							</tt:XRange>
							<tt:YRange>
								<tt:Min>-1</tt:Min>
								<tt:Max>1</tt:Max>
							</tt:YRange>
						</tt:Range>
					</tt:PanTiltLimits>
					<tt:ZoomLimits>
						<tt:Range>
							<tt:URI>http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace</tt:URI>
							<tt:XRange>
								<tt:Min>-1</tt:Min>
								<tt:Max>1</tt:Max>
							</tt:XRange>
						</tt:Range>
					</tt:ZoomLimits>
				</tt:PTZConfiguration>
			</trt:Profiles>
		</trt:GetProfilesResponse>
	</SOAP-ENV:Body>
</SOAP-ENV:Envelope>
''',
"http://www.onvif.org/ver20/ptz/wsdl/GetConfigurationOptions":'''
<?xml version='1.0' encoding='utf-8'?>
<soap-env:Envelope
	xmlns:soap-env="http://www.w3.org/2003/05/soap-envelope"
	xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
	xmlns:wsa="http://www.w3.org/2005/08/addressing">
	<soap-env:Header>
		<wsa:Action>http://www.onvif.org/ver20/ptz/wsdl/GetConfigurationOptions</wsa:Action>
		<wsa:MessageID>urn:uuid:d64f3d4e-4d2a-4caf-afb0-9230ddb9a755</wsa:MessageID>
		<wsa:To>http://127.0.0.1:1981/onvif/ptz</wsa:To>
		<wsse:Security
			xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
			<wsse:UsernameToken>
				<wsse:Username>admin</wsse:Username>
				<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">THrBWudzVUhWAIqGP9d5LqVKgAo=</wsse:Password>
				<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">JWQbV2RNlHE9C2Czn28oOA==</wsse:Nonce>
				<wsu:Created
					xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">2025-06-08T21:55:38+00:00
				</wsu:Created>
			</wsse:UsernameToken>
		</wsse:Security>
	</soap-env:Header>
	<soap-env:Body>
		<ns0:GetConfigurationOptions
			xmlns:ns0="http://www.onvif.org/ver20/ptz/wsdl">
			<ns0:ConfigurationToken>ptz0</ns0:ConfigurationToken>
		</ns0:GetConfigurationOptions>
	</soap-env:Body>
</soap-env:Envelope>
''',
"http://www.onvif.org/ver20/ptz/wsdl/GetPresets":'''
<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
	xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
	xmlns:SOAP-ENC="http://www.w3.org/2003/05/soap-encoding"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xmlns:xsd="http://www.w3.org/2001/XMLSchema"
	xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
	xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
	xmlns:chan="http://schemas.microsoft.com/ws/2005/02/duplex"
	xmlns:wsa5="http://www.w3.org/2005/08/addressing"
	xmlns:xmime="http://www.w3.org/2005/05/xmlmime"
	xmlns:xop="http://www.w3.org/2004/08/xop/include"
	xmlns:wsrfbf="http://docs.oasis-open.org/wsrf/bf-2"
	xmlns:tt="http://www.onvif.org/ver10/schema"
	xmlns:wstop="http://docs.oasis-open.org/wsn/t-1"
	xmlns:wsrfr="http://docs.oasis-open.org/wsrf/r-2"
	xmlns:tan="http://www.onvif.org/ver20/analytics/wsdl"
	xmlns:tdn="http://www.onvif.org/ver10/network/wsdl"
	xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
	xmlns:tev="http://www.onvif.org/ver10/events/wsdl"
	xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
	xmlns:c14n="http://www.w3.org/2001/10/xml-exc-c14n#"
	xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
	xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
	xmlns:wsc="http://schemas.xmlsoap.org/ws/2005/02/sc"
	xmlns:ds="http://www.w3.org/2000/09/xmldsig#"
	xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
	xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl"
	xmlns:tmd="http://www.onvif.org/ver10/deviceIO/wsdl"
	xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
	xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
	xmlns:ter="http://www.onvif.org/ver10/error"
	xmlns:tns1="http://www.onvif.org/ver10/topics"
	xmlns:trt2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tr2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tplt="http://www.onvif.org/ver10/plus/schema"
	xmlns:tpl="http://www.onvif.org/ver10/plus/wsdl"
	xmlns:ewsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:exsd="http://www.onvifext.com/onvif/ext/ver10/schema"
	xmlns:tnshik="http://www.hikvision.com/2011/event/topics"
	xmlns:hikwsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:hikxsd="http://www.onvifext.com/onvif/ext/ver10/schema">
	<SOAP-ENV:Header>
		<wsa5:MessageID>urn:uuid:de15225f-aabd-4294-802e-0a4f06e29779</wsa5:MessageID>
		<wsa5:To SOAP-ENV:mustUnderstand="true">http://192.168.1.253:80/onvif/ptz</wsa5:To>
		<wsa5:Action SOAP-ENV:mustUnderstand="true">http://www.onvif.org/ver20/ptz/wsdl/GetPresets</wsa5:Action>
		<wsse:Security>
			<wsse:UsernameToken>
				<wsse:Username>admin</wsse:Username>
				<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">pCoXIhvOQSp+Bqc5qyygFr5C4vc=</wsse:Password>
				<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">m1n6cBQstKqYFn1JAc0tcA==</wsse:Nonce>
				<wsu:Created>2025-06-08T21:55:38+00:00</wsu:Created>
			</wsse:UsernameToken>
		</wsse:Security>
	</SOAP-ENV:Header>
	<SOAP-ENV:Body>
		<tptz:GetPresetsResponse>
			<tptz:Preset token="1">
				<tt:Name>1</tt:Name>
			</tptz:Preset>
		</tptz:GetPresetsResponse>
	</SOAP-ENV:Body>
</SOAP-ENV:Envelope>
''',
"http://www.onvif.org/ver20/ptz/wsdl/ContinuousMove":'''
<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
	xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
	xmlns:SOAP-ENC="http://www.w3.org/2003/05/soap-encoding"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xmlns:xsd="http://www.w3.org/2001/XMLSchema"
	xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
	xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
	xmlns:chan="http://schemas.microsoft.com/ws/2005/02/duplex"
	xmlns:wsa5="http://www.w3.org/2005/08/addressing"
	xmlns:xmime="http://www.w3.org/2005/05/xmlmime"
	xmlns:xop="http://www.w3.org/2004/08/xop/include"
	xmlns:wsrfbf="http://docs.oasis-open.org/wsrf/bf-2"
	xmlns:tt="http://www.onvif.org/ver10/schema"
	xmlns:wstop="http://docs.oasis-open.org/wsn/t-1"
	xmlns:wsrfr="http://docs.oasis-open.org/wsrf/r-2"
	xmlns:tan="http://www.onvif.org/ver20/analytics/wsdl"
	xmlns:tdn="http://www.onvif.org/ver10/network/wsdl"
	xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
	xmlns:tev="http://www.onvif.org/ver10/events/wsdl"
	xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
	xmlns:c14n="http://www.w3.org/2001/10/xml-exc-c14n#"
	xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
	xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
	xmlns:wsc="http://schemas.xmlsoap.org/ws/2005/02/sc"
	xmlns:ds="http://www.w3.org/2000/09/xmldsig#"
	xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
	xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl"
	xmlns:tmd="http://www.onvif.org/ver10/deviceIO/wsdl"
	xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
	xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
	xmlns:ter="http://www.onvif.org/ver10/error"
	xmlns:tns1="http://www.onvif.org/ver10/topics"
	xmlns:trt2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tr2="http://www.onvif.org/ver20/media/wsdl"
	xmlns:tplt="http://www.onvif.org/ver10/plus/schema"
	xmlns:tpl="http://www.onvif.org/ver10/plus/wsdl"
	xmlns:ewsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:exsd="http://www.onvifext.com/onvif/ext/ver10/schema"
	xmlns:tnshik="http://www.hikvision.com/2011/event/topics"
	xmlns:hikwsd="http://www.onvifext.com/onvif/ext/ver10/wsdl"
	xmlns:hikxsd="http://www.onvifext.com/onvif/ext/ver10/schema">
	<SOAP-ENV:Header>
		<wsa5:MessageID>urn:uuid:e6a20ac2-f771-43f5-b364-77becfe3c0d2</wsa5:MessageID>
		<wsa5:To SOAP-ENV:mustUnderstand="true">http://192.168.1.253:80/onvif/ptz</wsa5:To>
		<wsa5:Action SOAP-ENV:mustUnderstand="true">http://www.onvif.org/ver20/ptz/wsdl/ContinuousMove</wsa5:Action>
		<wsse:Security>
			<wsse:UsernameToken>
				<wsse:Username>admin</wsse:Username>
				<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">keJSs3pC0udf+9afM2UjwFhH0uw=</wsse:Password>
				<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">SX08nwntn+2HnVLSa0CSrA==</wsse:Nonce>
				<wsu:Created>2025-06-08T21:55:38+00:00</wsu:Created>
			</wsse:UsernameToken>
		</wsse:Security>
	</SOAP-ENV:Header>
	<SOAP-ENV:Body>
		<tptz:ContinuousMoveResponse></tptz:ContinuousMoveResponse>
	</SOAP-ENV:Body>
</SOAP-ENV:Envelope>

'''
}

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 1981         # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 1024

# simulate an ONVIF device locally
def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)  # Listen for one incoming connection at a time
        print(f"SOAP ONVIF server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()  # Accept a new connection
            with conn:
                #buffer = b""  # Buffer to store incoming data
                #while True:
                #    data = conn.recv(BUFFER_SIZE)  # Read data from the socket
                #    if not data:
                #        break  # Connection closed
                #    buffer += data

                print(f"SOAP ONVIF connected by {addr}")
                #request_data = buffer #
                request_data = conn.recv(4096).decode('utf-8')  # Receive data from the client

                print(f"SOAP ONVIF received request:\n{request_data}")

                request_lines = request_data.splitlines()

                soap_action = None
                soap_service = None
                for line in request_lines:
                    if "SOAPAction:" in line:
                        quoted_strings = re.findall(r'"(.*?)"', line)
                        if len(quoted_strings)>0:
                            soap_action = quoted_strings[0]
                            soap_service_strings = re.findall(r'[^\/]+(?=\/?$|\?|#)', quoted_strings[0])
                            if len(soap_service_strings)>0:
                                soap_service = soap_service_strings[0]
                                break

                assert soap_action is not None
                assert soap_service is not None


                response_body = ONVIF_SOAP_RESPONSES.get(soap_action)

                assert response_body is not None
                print(f"soap response: {soap_action}")
                http_response = [
                    "HTTP/1.1 200 OK",
                    f"Content-Type: application/soap+xml; charset=utf-8; action=\"{soap_action}\"",
                    "Server: gSOAP/2.8",
                    "Access-Control-Allow-Origin: *",
                    f"Content-Length: {len(response_body)}",
                    "Connection: close",
                    "\r\n",
                    f"{response_body}"
                ]
                bytes_to_send = "\r\n".join(http_response).encode('utf-8')
                total_sent = 0
                while total_sent < len(bytes_to_send):
                    segment = bytes_to_send[total_sent : max(total_sent + BUFFER_SIZE,len(bytes_to_send))]
                    sent = conn.send(segment)
                    if sent == 0:
                        raise RuntimeError("Socket connection broken")
                    total_sent += sent
                print(f"sent: {total_sent}")

def get_capabilities() -> str:
    return "{}"

def test_workflow_with_onvif(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=ONVIF_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
        }
    )

    output = result[0].get("output")
    assert output is not None

    assert output["tracker_id"]>0
    assert output["predictions"]

    time.sleep(5)
    assert output["seeking"]
