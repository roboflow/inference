import json
import os
import socket
import warnings

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "9999"))
BUFFER_SIZE = 65535


def main() -> None:
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind((HOST, PORT))
    try:
        while True:
            message, _ = udp_socket.recvfrom(BUFFER_SIZE)
            if len(message) > BUFFER_SIZE:
                warnings.warn("Message exceeds buffer size")
                continue
            try:
                decoded_message = message.decode("utf-8")
                parsed_message = json.loads(decoded_message)
                print(parsed_message)
            except UnicodeDecodeError:
                warnings.warn("Failed to decode message as UTF-8")
                continue
            except json.JSONDecodeError:
                warnings.warn("Failed to parse message as JSON")
                continue
    finally:
        udp_socket.close()


if __name__ == '__main__':
    main()