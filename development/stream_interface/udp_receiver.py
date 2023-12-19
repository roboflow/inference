import json
import os
import socket

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "9999"))
BUFFER_SIZE = 65535


def main() -> None:
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind((HOST, PORT))
    try:
        while True:
            message, _ = udp_socket.recvfrom(BUFFER_SIZE)
            decoded_message = message.decode("utf-8")
            parsed_message = json.loads(decoded_message)
            print(parsed_message)
    finally:
        udp_socket.close()


if __name__ == '__main__':
    main()
