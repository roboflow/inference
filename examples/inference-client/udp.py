import argparse
import json
import socket
import time

parser = argparse.ArgumentParser(description="Receive stream detections over UDP.")

parser.add_argument(
    "--port", type=int, required=True, help="Port to listen on.", default=12345
)

arguments = parser.parse_args()


def start_udp_server(ip: str, port: int):
    fps_array = []

    # Create a datagram (UDP) socket
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Bind to the given IP address and port
    UDPClientSocket.bind((ip, port))

    print(f"UDP server up and listening on http://{ip}:{port}")

    # Listen for incoming datagrams
    while True:
        t0 = time.time()

        bytesAddressPair = UDPClientSocket.recvfrom(1024)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        clientMsg = json.loads(message)
        clientIP = "Client IP Address:{}".format(address)

        print(clientMsg)
        print(clientIP)

        t = time.time() - t0
        fps_array.append(1 / t)
        fps_array[-150:]
        fps_average = sum(fps_array) / len(fps_array)
        print("AVERAGE FPS: " + str(fps_average))


if __name__ == "__main__":
    start_udp_server("localhost", arguments.port)
