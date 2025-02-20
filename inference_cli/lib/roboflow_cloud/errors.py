class RoboflowCloudCommandError(Exception):
    pass


class RetryError(RoboflowCloudCommandError):
    pass


class RFAPICallError(RoboflowCloudCommandError):
    pass


class UnauthorizedRequestError(RFAPICallError):
    pass
