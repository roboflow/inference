class CommunicationProtocolError(Exception):
    pass


class MessageToBigError(CommunicationProtocolError):
    pass


class MalformedHeaderError(CommunicationProtocolError):
    pass


class TransmissionChannelClosed(CommunicationProtocolError):
    pass


class MalformedPayloadError(CommunicationProtocolError):
    pass
